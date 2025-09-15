import torch

def bilinear_interpolate(feature_map, x, y):
    bs_heads, embed_dims, H, W = feature_map.shape

    x = (x + 1) * (W - 1) / 2
    y = (y + 1) * (H - 1) / 2

    x = torch.clamp(x, 0, W - 1)
    y = torch.clamp(y, 0, H - 1)

    x0 = torch.floor(x).long()
    x1 = torch.clamp(x0 + 1, 0, W - 1)
    y0 = torch.floor(y).long()
    y1 = torch.clamp(y0 + 1, 0, H - 1)

    wa = ((x1.float() - x) * (y1.float() - y)).unsqueeze(1)
    wb = ((x - x0.float()) * (y1.float() - y)).unsqueeze(1)
    wc = ((x1.float() - x) * (y - y0.float())).unsqueeze(1)
    wd = ((x - x0.float()) * (y - y0.float())).unsqueeze(1)

    bs_heads_idx = torch.arange(bs_heads, device=feature_map.device).view(-1, 1, 1, 1)
    embed_idx = torch.arange(embed_dims, device=feature_map.device).view(1, -1, 1, 1)

    Ia = feature_map[bs_heads_idx, embed_idx, y0.unsqueeze(1), x0.unsqueeze(1)]
    Ib = feature_map[bs_heads_idx, embed_idx, y0.unsqueeze(1), x1.unsqueeze(1)]
    Ic = feature_map[bs_heads_idx, embed_idx, y1.unsqueeze(1), x0.unsqueeze(1)]
    Id = feature_map[bs_heads_idx, embed_idx, y1.unsqueeze(1), x1.unsqueeze(1)]

    interpolated = wa * Ia + wb * Ib + wc * Ic + wd * Id
    
    return interpolated

def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """
    Versione CPU di multi-scale deformable attention senza grid_sample.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    sampling_grids = 2 * sampling_locations - 1
    
    sampling_value_list = []
    
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)

        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        x_coords = sampling_grid_l_[:, :, :, 0] 
        y_coords = sampling_grid_l_[:, :, :, 1] 

        sampling_value_l_ = bilinear_interpolate(value_l_, x_coords, y_coords)
        
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)

    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    
    return output.transpose(1, 2).contiguous()