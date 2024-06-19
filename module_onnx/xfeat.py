import os
import numpy as np
import torch

from module_onnx.model import *
from module_onnx.interpolator import InterpolateSparse2d
import torch.nn.functional as F

class XFeat(nn.Module):
    """
		Implements the inference module for XFeat.
		It supports inference for both sparse and semi-dense feature extraction & matching.
	"""

    def __init__(self, weights=None, top_k=4096, multiscale=False):
        super().__init__()
        # self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dev = torch.device('cpu')
        self.net = XFeatModel().to(self.dev)
        self.top_k = top_k
        self.multiscale = multiscale

        if weights is not None:
            if isinstance(weights, str):
                print('loading weights from: ' + weights)
                self.net.load_state_dict(torch.load(weights, map_location=self.dev), strict=False)
            else:
                self.net.load_state_dict(weights)

        self.interpolator = InterpolateSparse2d('bicubic')

    @torch.inference_mode()
    def detectAndCompute(self, x):
        """
            Compute sparse keypoints & descriptors. Supports batched mode.

            input:
                x -> torch.Tensor(1, C, H, W): grayscale or rgb image
                top_k -> int: keep best k features
            return:
                List[Dict]:
                    'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                    'scores'       ->   torch.Tensor(N,): keypoint scores
                    'descriptors'  ->   torch.Tensor(N, 64): local features
        """
        # image.shape == (1, 3, H, W)
        x, rh1, rw1 = self.preprocess_tensor(x)
        _, _, _H1, _W1 = x.shape

        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        # Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=0.05, kernel_size=5)

        # Compute reliability scores
        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        # Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :self.top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :self.top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :self.top_k]

        # Interpolate descriptors at kpts positions
        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)

        # L2-Normalize
        feats = F.normalize(feats, dim=-1)

        # Correct kpt scale
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)
        valid = scores > 0
        return {'keypoints': mkpts[valid],
                'descriptors': feats[valid],
                'scores': scores[valid]}

    @torch.inference_mode()
    def detectAndComputeDense(self, x):
        """
            Compute dense *and coarse* descriptors. Supports batched mode.

            input:
                x -> torch.Tensor(B, C, H, W): grayscale or rgb image
                top_k -> int: keep best k features
            return: features sorted by their reliability score -- from most to least
                List[Dict]:
                    'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
                    'scales'       ->   torch.Tensor(top_k,): extraction scale
                    'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
        """
        if self.multiscale:
            print("TODO: Multiscale export")
            exit(-1)
            mkpts, sc, feats = self.extract_dualscale(x, self.top_k)
        else:
            mkpts, feats = self.extractDense(x)
            sc = torch.ones(mkpts.shape[:1], device=mkpts.device)

        return {'keypoints': mkpts,
                'descriptors': feats,
                'scales': sc}

    def create_xy(self, h, w, dev):
        y, x = torch.meshgrid(torch.arange(h, device=dev),
                              torch.arange(w, device=dev), indexing='ij')
        xy = torch.cat([x[..., None], y[..., None]], -1).reshape(-1, 2).float()
        return xy

    def extractDense(self, x):

        x, rh1, rw1 = self.preprocess_tensor(x)
        # x, self.net.scales[0], self.net.scales[1] = self.preprocess_tensor(x)
        M1, K1, H1 = self.net(x)
        _, C, _H1, _W1 = M1.shape
        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8)

        M1 = M1[0].permute(1, 2, 0).flatten(0, 1)  # 1, H*W, C
        H1 = H1[0].permute(1, 2, 0).flatten(0)  # 1, H*W

        # _, top_k = torch.topk(H1, k = min(H1.shape[1], top_k), dim=-1)
        _, top_k = torch.topk(H1, torch.min(torch.from_numpy(np.array([H1.shape[0], self.top_k]))), dim=-1)

        #feats = torch.gather(M1, 1, top_k[..., None].expand(-1, -1, 64))
        #mkpts = torch.gather(xy1, 1, top_k[..., None].expand(-1, -1, 2))
        feats = torch.gather(M1, 0, top_k[..., None].expand(-1, 64))
        mkpts = torch.gather(xy1, 0, top_k[..., None].expand(-1, 2))

        # Avoid warning of torch.tensor being treated as a constant when exporting to ONNX
        # mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, -1)
        mkpts[..., 0] = mkpts[..., 0] * rw1
        mkpts[..., 1] = mkpts[..., 1] * rh1

        return mkpts, feats

    @torch.inference_mode()
    def match_xfeat(self, img1, img2):
        """
            Simple extractor and MNN matcher.
            For simplicity, it does not support batched mode due to possibly different number of kpts.
            input:
                img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                top_k -> int: keep best k features
            returns:
                mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        # img1 = self.parse_input(img1)
        # img2 = self.parse_input(img2)

        out1 = self.detectAndCompute(img1)
        out2 = self.detectAndCompute(img2)

        idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'])
        return out1['keypoints'][idxs0], out2['keypoints'][idxs1]

    @torch.inference_mode()
    def match_xfeat_star(self, img1, img2):
        """
            Simple extractor and MNN matcher.
            For simplicity, it does not support batched mode due to possibly different number of kpts.
            input:
                img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                top_k -> int: keep best k features
            returns:
                mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        out1 = self.detectAndComputeDense(img1)
        out2 = self.detectAndComputeDense(img2)
        idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'])

        return self.refine_matches( out1["keypoints"], out1["descriptors"], idxs0,
                                    out2["keypoints"], out2["descriptors"], idxs1,
                                    out1["scales"])


    @torch.inference_mode()
    def match(self, feats1, feats2, min_cossim=-1):

        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()

        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(match12.shape[0], device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1

    @torch.inference_mode()
    def match_onnx(self, mkpts0, feats0, mkpts1, feats1):
        idx0, idx1 = self.match(feats0, feats1, min_cossim=-1)
        return mkpts0[idx0], mkpts1[idx1]

    @torch.inference_mode()
    def match_star_onnx(self, mkpts0, feats0, mkpts1, feats1, sc0):
        idx0, idx1 = self.match(feats0, feats1, min_cossim=0.82)
        # Refine coarse matches
        return self.refine_matches(mkpts0, feats0, idx0, mkpts1, feats1, idx1, sc0, fine_conf=0.25)

    def subpix_softmax2d(self, heatmaps, temp=3):
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H * W), -1).view(-1, H, W)
        x, y = torch.meshgrid(torch.arange(H, device=heatmaps.device), torch.arange(W, device=heatmaps.device),
                              indexing='ij')
        x = x - (W // 2)
        y = y - (H // 2)

        coords_x = (x[None, ...] * heatmaps)
        coords_y = (y[None, ...] * heatmaps)
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H * W, 2)
        coords = coords.sum(1)

        return coords

    def refine_matches(self, mkpts0, d0, idx0, mkpts1, d1, idx1, sc0, fine_conf=0.25):
        feats1 = d0[idx0]  # [idx0_b[:, 0], idx0_b[:, 1]]
        feats2 = d1[idx1]  # [idx1_b[:, 0], idx1_b[:, 1]]
        mkpts_0 = mkpts0[idx0]  # [idx0_b[:, 0], idx0_b[:, 1]]
        mkpts_1 = mkpts1[idx1]  # [idx1_b[:, 0], idx1_b[:, 1]]
        sc0 = sc0[idx0]  # [idx0_b[:, 0], idx0_b[:, 1]]

        # Compute fine offsets
        offsets = self.net.fine_matcher(torch.cat([feats1, feats2], dim=-1))
        conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))

        mkpts_0 += offsets * (sc0[:, None])  # *0.9 #* (sc0[:,None])
        mkpts_1 += offsets * (sc0[:, None])  # *0.9 #* (sc0[:,None])

        mask_good = conf > fine_conf
        mkpts_0 = mkpts_0[mask_good]
        mkpts_1 = mkpts_1[mask_good]

        # match_mkpts = torch.cat([mkpts_0, mkpts_1], dim=-1)
        # batch_index = idx0[mask_good]  # idx0_b[mask_good, 0]
        return mkpts_0, mkpts_1

    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        x = x.to(self.dev).float()

        H, W = x.shape[-2:]
        _H, _W = (H // 32) * 32, (W // 32) * 32
        rh, rw = H / _H, W / _W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

    def NMS(self, x, threshold=0.05, kernel_size=5):
        _, _, H, W = x.shape
        local_max = F.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            return_indices=False,
        )
        pos = (x == local_max) & (x > threshold)
        return pos.squeeze().nonzero().flip(-1).reshape(1, -1, 2)



