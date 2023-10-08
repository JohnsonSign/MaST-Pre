import os
import sys
import time
import torch
import math
import numpy as np

# Useful constants.
PST_PI = 3.1415926535897932384626433832795
PST_RAD_45 = 0.78539816339744830961566084581988
PST_RAD_90 = 1.5707963267948966192313216916398
PST_RAD_135 = 2.3561944901923449288469825374596
PST_RAD_180 = PST_PI
PST_RAD_360 = 6.283185307179586476925286766558
PST_RAD_PI_7_8 = 2.7488935718910690836548129603691


def getSHOTLocalRF(centralPoints, neighs, radius, nnidx):
    displacement = (neighs - centralPoints.unsqueeze(2))                     # (B, N', k, 3)
    dists = torch.sqrt(torch.sum(torch.pow(displacement, 2), dim=-1))        # (B, N', k)

    w = (radius - dists).clamp(min=0.0)                                      # (B, N', k)

    # idx mask
    idx_mask = (nnidx - nnidx[:,:,0].unsqueeze(-1)).bool()
    idx_mask[:,:,0] = True 
    idx_bool = torch.sum(nnidx, dim=-1).bool()                               # for all nnidx are 0
    idx_mask[~idx_bool] = False

    w = w * idx_mask
    sumw = torch.sum(w, dim=-1, keepdim=True)                                # (B, N', 1)

    weightDis = torch.einsum('bnkc,bnk->bnkc', displacement, torch.sqrt(w))  # (B, N', k, 3)

    M = torch.einsum('bnkc,bnkd->bncd', weightDis, weightDis)                # (B, N', 3, 3)
    M = M / (sumw.unsqueeze(2) + 1e-10)

    # Eigenvalue decomposition
    eVal, eVec = torch.symeig(M.cpu(), eigenvectors=True)                    # (B, N', 3) (B, N', 3, 3) ascending order
    # assert((eVal[:,:,0]<=eVal[:,:,1]).all() and (eVal[:,:,1]<=eVal[:,:,2]).all()), "[ERROR] eigenvalues should be ascending order."

    ref_Z = eVec[:,:,:,0].cuda()  # min  (B, N', 3)
    ref_X = eVec[:,:,:,2].cuda()  # max  (B, N', 3)

    # Disambiguate x and z
    xInFeatRef = torch.einsum('bnkc,bnc->bnk', displacement, ref_X)          # (B, N', k)
    zInFeatRef = torch.einsum('bnkc,bnc->bnk', displacement, ref_Z)          # (B, N', k)

    disX_pos = ((xInFeatRef * idx_mask) > 0) 
    disX_pos = torch.sum(disX_pos, dim=-1, keepdim=True)                     # (B, N', 1)
    disZ_pos = ((zInFeatRef * idx_mask) > 0) 
    disZ_pos = torch.sum(disZ_pos, dim=-1, keepdim=True)                     # (B, N', 1)

    nk = torch.sum(idx_mask, dim=-1, keepdim=True)
    ref_X = torch.where(disX_pos < (nk-disX_pos), -ref_X, ref_X)
    ref_Z = torch.where(disZ_pos < (nk-disZ_pos), -ref_Z, ref_Z)

    ref_Y = torch.cross(ref_Z, ref_X, dim=-1)
    
    return ref_X, ref_Y, ref_Z                                # (B, N', k)


def getDescribe(centralPoints, neighs,
                SHOTradius,
                useInterpolation, useNormalization, 
                nnidx
    ):
    # idx mask
    idx_mask = (nnidx - nnidx[:,:,0].unsqueeze(-1)).bool()
    idx_mask[:,:,0] = True 
    idx_bool = torch.sum(nnidx, dim=-1).bool()                                    # for all nnidx are 0
    idx_mask[~idx_bool] = False                                                   # (B, N', k)
    
    BBB, NNN, KKK, _ = neighs.shape

    displacement = (neighs - centralPoints.unsqueeze(2))                          # (B, N', k, 3)
    dists = torch.sqrt(torch.sum(torch.pow(displacement, 2), dim=-1))             # (B, N', k)

    xInFeatRef = displacement[:,:,:,0]
    yInFeatRef = displacement[:,:,:,1]
    zInFeatRef = displacement[:,:,:,2]
    
    # The location codeing of 32 volumes
    bit4 = torch.where(((yInFeatRef > 0) | ((yInFeatRef == 0.0) & (xInFeatRef < 0))), 1, 0)          
    bit3 = torch.where(((xInFeatRef > 0) | ((xInFeatRef == 0.0) & (yInFeatRef > 0))), 1-bit4, bit4)  
    
    desc_index = ((bit4 * 4) + (bit3 * 2))
    desc_index = torch.where(zInFeatRef > 0, desc_index+1, desc_index)

    # For interpolation
    lenDescriptor = getDescriptorLength()

    if not useInterpolation:
        desc_index = desc_index.flatten().unsqueeze(-1)                           # (B*N'*k, 1)

        desc_index_one_hot = torch.zeros_like(desc_index).repeat(1,lenDescriptor)
        descriptor = desc_index_one_hot.scatter_(1, desc_index, 1)                # (B*N'*k, L)

        descriptor = descriptor.reshape(BBB, NNN, KKK, lenDescriptor)             # (B, N', k, L)
        descriptor[~idx_mask] = 0
        descriptor = torch.sum(descriptor, dim=2)                                 # (B, N', L)

    elif useInterpolation:
        # 3. Interpolation on the inclination (adjacent vertical volumes)
        inclinationCos = zInFeatRef / (dists + 1E-15)                                   # (B, N', k)
        inclinationCos = inclinationCos.clamp(max=1.0, min=-1.0)
        
        inclination = torch.acos(inclinationCos)                                        # (B, N', k)
        # assert((inclination >= 0.0).all() and (inclination <= PST_PI).all())

        lowerHemisphere = ((inclination > PST_RAD_90) | ((abs(inclination - PST_RAD_90)<1e-6) & (zInFeatRef <= 0)))

        inclinationDistance = torch.where(lowerHemisphere,
                                            (inclination - PST_RAD_135) / PST_RAD_90,
                                            (inclination - PST_RAD_45) / PST_RAD_90)

        lowerSphere = (lowerHemisphere & (inclination <= PST_RAD_135))
        higherSphere = (~lowerHemisphere & (inclination >= PST_RAD_45))

        intWeight = torch.where(lowerHemisphere & (inclination > PST_RAD_135),
                                (1 - inclinationDistance).double(), 0.0)
        intWeight = torch.where(lowerSphere,
                                intWeight + (1 + inclinationDistance), intWeight)
        intWeight = torch.where(~lowerHemisphere & (inclination < PST_RAD_45),
                                intWeight + (1 + inclinationDistance), intWeight)
        intWeight = torch.where(higherSphere,
                                intWeight + (1 - inclinationDistance), intWeight)

        # 3.1 PST_RAD_135
        inter_index = torch.where(lowerSphere, (desc_index + 1), 0)
        inter_index = inter_index.flatten().unsqueeze(-1)                               # (B*N'*k, 1)
        # assert((inter_index >= 0).all() and (inter_index < lenDescriptor).all())

        inter_weight = torch.where(lowerSphere, (-inclinationDistance).double(), 0.0)   # (B, N', k)
        inter_weight = inter_weight.flatten()

        inter_index_one_hot = torch.zeros_like(inter_index).repeat(1,lenDescriptor)     # (B*N'*k, L)
        inter_index_one_hot = inter_index_one_hot.scatter_(1, inter_index, 1)

        descriptor = torch.einsum('il,i->il', inter_index_one_hot, inter_weight)       # (B*N'*k, L)

        # 3.2 PST_RAD_45
        inter_index = torch.where(higherSphere, (desc_index - 1), 0)
        inter_index = inter_index.flatten().unsqueeze(-1)                               # (B*N'*k, 1)
        # assert((inter_index >= 0).all() and (inter_index < lenDescriptor).all())

        inter_weight = torch.where(higherSphere, (inclinationDistance).double(), 0.0)   # (B, N', k)
        inter_weight = inter_weight.flatten()

        inter_index_one_hot = torch.zeros_like(inter_index).repeat(1,lenDescriptor)     # (B*N'*k, L)
        inter_index_one_hot = inter_index_one_hot.scatter_(1, inter_index, 1)

        descriptor += torch.einsum('il,i->il', inter_index_one_hot, inter_weight)       # (B*N'*k, L)

        # 4. Interpolation on the azimuth (adjacent horizontal volumes)
        x_y_InFeatRef = ((yInFeatRef != 0.0) | (xInFeatRef != 0.0))
        sel = torch.where(x_y_InFeatRef, (desc_index // 2), 0)

        angularSectorSpan = PST_RAD_90
        angularSectorStart = (-PST_RAD_135)

        azimuth = torch.where(x_y_InFeatRef, torch.atan2(yInFeatRef, xInFeatRef).double(), 0.0)

        azimuthDistance = torch.where(x_y_InFeatRef,
            ((azimuth - (angularSectorStart + angularSectorSpan * sel)) / angularSectorSpan).double(), 0.0)
        # assert((azimuthDistance < (0.5+1e-15)).all() and (azimuthDistance > (-0.5-1e-15)).all())

        azimuthDistance = azimuthDistance.clamp(max=0.5, min=-0.5)
        xyRef_azimuthDisB0 = (x_y_InFeatRef & (azimuthDistance > 0))
        xyRef_azimuthDisS0 = (x_y_InFeatRef & (azimuthDistance <= 0))
        intWeight = torch.where(xyRef_azimuthDisB0, (intWeight + (1 - azimuthDistance)).double(), intWeight.double())
        intWeight = torch.where(xyRef_azimuthDisS0, (intWeight + (1 + azimuthDistance)).double(), intWeight.double())

        # 4.1
        inter_index = torch.where(xyRef_azimuthDisB0,
                                ((desc_index + 2) % lenDescriptor), 0)
        inter_index = inter_index.flatten().unsqueeze(-1)                               # (B*N'*k, 1)
        # assert((inter_index >= 0).all() and (inter_index < lenDescriptor).all())

        inter_weight = torch.where(xyRef_azimuthDisB0, (azimuthDistance).double(), 0.0) # (B, N', k)
        inter_weight = inter_weight.flatten()

        inter_index_one_hot = torch.zeros_like(inter_index).repeat(1,lenDescriptor)     # (B*N'*k, L)
        inter_index_one_hot = inter_index_one_hot.scatter_(1, inter_index, 1)

        descriptor += torch.einsum('il,i->il', inter_index_one_hot, inter_weight)       # (B*N'*k, L)

        # 4.2
        inter_index = torch.where(xyRef_azimuthDisS0,
                                ((desc_index - 2 + lenDescriptor) % lenDescriptor), 0)
        inter_index = inter_index.flatten().unsqueeze(-1)             # (B*N'*k, 1)
        # assert((inter_index >= 0).all() and (inter_index < lenDescriptor).all())

        inter_weight = torch.where(xyRef_azimuthDisS0,(-azimuthDistance).double(), 0.0) # (B, N', k)
        inter_weight = inter_weight.flatten()

        inter_index_one_hot = torch.zeros_like(inter_index).repeat(1,lenDescriptor)     # (B*N'*k, L)
        inter_index_one_hot = inter_index_one_hot.scatter_(1, inter_index, 1)

        descriptor += torch.einsum('il,i->il', inter_index_one_hot, inter_weight)       # (B*N'*k, L)

        # add current point
        inter_index = desc_index.flatten().unsqueeze(-1)                                # (B*N'*k, 1)
        # assert((inter_index >= 0).all() and (inter_index < lenDescriptor).all())

        inter_index_one_hot = torch.zeros_like(inter_index).repeat(1,lenDescriptor)     # (B*N'*k, L)
        inter_index_one_hot = inter_index_one_hot.scatter_(1, inter_index, 1)

        intWeight = intWeight.flatten()                                                 # (B*N'*k)
        descriptor += torch.einsum('il,i->il', inter_index_one_hot, intWeight)          # (B*N'*k, L)

        descriptor = descriptor.reshape(BBB, NNN, KKK, lenDescriptor)                   # (B, N', k, L)
        descriptor[~idx_mask] = 0
        descriptor = torch.sum(descriptor, dim=2)                                       # (B, N', L)

    if useNormalization:
        accNorm = torch.sqrt(torch.sum(torch.pow(descriptor, 2), dim=-1, keepdim=True)) # (B, N', 1)
        descriptor = descriptor / (accNorm + 1E-15)                                     # (B, N', L)

    return descriptor.float() # (B, N', L)



def getDescriptorLength(elevation_divisions=2,
                        azimuth_divisions=4
    ):
    lenDescriptor = elevation_divisions * azimuth_divisions
    return lenDescriptor


