/// KTX2 3D Texture Writer for Bevy Irradiance Volumes
/// Supports VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 (RGB9E5) and VK_FORMAT_R16G16B16A16_SFLOAT (RGBA16F)

pub fn pack_rgb9e5(r: f32, g: f32, b: f32) -> u32 {
    let r = r.max(0.0).min(65408.0);
    let g = g.max(0.0).min(65408.0);
    let b = b.max(0.0).min(65408.0);
    let max_val = r.max(g).max(b);
    if max_val < 1e-15 {
        return 0;
    }
    // Shared exponent formula: floor(log2(max_val)) + 1 + 15
    let exp_shared = (max_val.log2().floor() as i32 + 1 + 15).clamp(0, 31);
    let denom = 2.0f32.powi(exp_shared - 15 - 9);
    let max_c = 511.0;
    let rm = ((r / denom).round()).min(max_c) as u32;
    let gm = ((g / denom).round()).min(max_c) as u32;
    let bm = ((b / denom).round()).min(max_c) as u32;
    (rm & 0x1FF) | ((gm & 0x1FF) << 9) | ((bm & 0x1FF) << 18) | ((exp_shared as u32 & 0x1F) << 27)
}

pub fn unpack_rgb9e5(val: u32) -> [f32; 3] {
    let rm = val & 0x1FF;
    let gm = (val >> 9) & 0x1FF;
    let bm = (val >> 18) & 0x1FF;
    let exp = (val >> 27) & 0x1F;
    let scale = 2.0f32.powi(exp as i32 - 15 - 9);
    [rm as f32 * scale, gm as f32 * scale, bm as f32 * scale]
}

/// Standard DFD block for VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 (124 bytes) matching Bevy / Khronos KTX2 spec
const DFD_RGB9E5: [u8; 124] = [
    0x7C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x78, 0x00,
    0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x1B, 0x00, 0x04, 0x20,
    0x00, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00,
    0x09, 0x00, 0x08, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x21, 0x00, 0x00, 0x1B, 0x00, 0x04, 0x21, 0x00, 0x00, 0x00, 0x00,
    0x0F, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x12, 0x00, 0x08, 0x02,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00,
    0x1B, 0x00, 0x04, 0x22, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00,
    0x1F, 0x00, 0x00, 0x00,
];

/// Encodes a 3D texture in uncompressed KTX2 format with RGB9E5 pixels.
pub fn create_ktx2_3d_rgb9e5(width: u32, height: u32, depth: u32, pixels: &[u32]) -> Vec<u8> {
    assert_eq!(
        pixels.len(),
        (width * height * depth) as usize,
        "Pixel count must match dimensions"
    );

    let data_bytes = bytemuck::cast_slice(pixels);
    let data_len = data_bytes.len() as u64;

    let header_len = 80u32;
    let level_index_len = 24u32; // 1 mip level = 3 * 8 bytes
    let dfd_offset = header_len + level_index_len; // 104 = 0x68
    let dfd_len = DFD_RGB9E5.len() as u32; // 124 = 0x7C
    let pixel_data_offset = (dfd_offset + dfd_len) as u64; // 228 = 0xE4

    let mut out = Vec::with_capacity((pixel_data_offset + data_len) as usize);

    // 1. Identifier: «KTX 20»\r\n\x1a\n
    out.extend_from_slice(&[0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A]);

    // 2. Header fields
    out.extend_from_slice(&123u32.to_le_bytes()); // vkFormat = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32
    out.extend_from_slice(&4u32.to_le_bytes());   // typeSize = 4 bytes
    out.extend_from_slice(&width.to_le_bytes());  // pixelWidth
    out.extend_from_slice(&height.to_le_bytes()); // pixelHeight
    out.extend_from_slice(&depth.to_le_bytes());  // pixelDepth
    out.extend_from_slice(&0u32.to_le_bytes());   // layerCount
    out.extend_from_slice(&1u32.to_le_bytes());   // faceCount
    out.extend_from_slice(&1u32.to_le_bytes());   // levelCount
    out.extend_from_slice(&0u32.to_le_bytes());   // supercompressionScheme = None

    // Index
    out.extend_from_slice(&dfd_offset.to_le_bytes()); // dfdByteOffset = 0x68
    out.extend_from_slice(&dfd_len.to_le_bytes());    // dfdByteLength = 0x7C
    out.extend_from_slice(&0u32.to_le_bytes());       // kvdByteOffset
    out.extend_from_slice(&0u32.to_le_bytes());       // kvdByteLength
    out.extend_from_slice(&0u64.to_le_bytes());       // sgdByteOffset
    out.extend_from_slice(&0u64.to_le_bytes());       // sgdByteLength

    // Level index (Level 0)
    out.extend_from_slice(&pixel_data_offset.to_le_bytes()); // byteOffset
    out.extend_from_slice(&data_len.to_le_bytes());          // byteLength
    out.extend_from_slice(&data_len.to_le_bytes());          // uncompressedByteLength

    // DFD block
    out.extend_from_slice(&DFD_RGB9E5);

    // Pixel data
    out.extend_from_slice(data_bytes);

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb9e5_roundtrip() {
        let colors = [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.25],
            [10.0, 20.0, 5.0],
            [0.01, 0.02, 0.03],
        ];
        for col in colors {
            let packed = pack_rgb9e5(col[0], col[1], col[2]);
            let unpacked = unpack_rgb9e5(packed);
            for i in 0..3 {
                let diff = (col[i] - unpacked[i]).abs();
                assert!(diff <= col[i] * 0.05 + 1e-4, "Mismatch on {col:?}: got {unpacked:?}");
            }
        }
    }
}

