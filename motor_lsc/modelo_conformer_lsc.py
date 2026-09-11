"""
=============================================================================
ARQUITECTURA CONFORMER MULTI-STREAM CON FUSIÓN POR ATENCIÓN CRUZADA (LSC)
=============================================================================
Diseñado específicamente para Lengua de Señas Colombiana (LSC):
1. Tres Streams Especializados:
   - Stream Manos: Conformer Block (1D Depthwise Conv + MHSA + Macaron FFN)
   - Stream Rostro: Conv1D Temporal + Auto-atención de NMMs (64 pts gramaticales)
   - Stream Torso: Linear + Spatial Anchor Conv (33 pts esternal)
2. Fusión por Atención Cruzada (Cross-Attention Fusion):
   - Las Manos (Articulador Principal Q) interrogan activamente al Rostro
     y Torso (K, V) para resolver ambigüedades posturales y gramaticales.
3. Doble Salida:
   - Cabezal CTC para Reconocimiento Continuo de secuencias de oraciones.
   - Cabezal de Pooling para Clasificación de Señas Aisladas.
4. Presupuesto Mobile: < 10M parámetros, exportable a ONNX y TFLite INT8/FP16.
=============================================================================
"""

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_DISPONIBLE = True
except ImportError:
    TORCH_DISPONIBLE = False

if TORCH_DISPONIBLE:

    class ModuloConvolucionConformer(nn.Module):
        """
        Módulo de Convolución 1D del Conformer:
        LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv1D -> BatchNorm -> SiLU -> Pointwise -> Dropout
        """
        def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
            super().__init__()
            self.layer_norm = nn.LayerNorm(d_model)
            self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
            self.glu = nn.GLU(dim=1)
            self.depthwise_conv = nn.Conv1d(
                d_model, d_model, kernel_size=kernel_size,
                padding=kernel_size // 2, groups=d_model
            )
            self.batch_norm = nn.BatchNorm1d(d_model)
            self.activation = nn.SiLU()
            self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [B, T, D]
            residual = x
            x = self.layer_norm(x)
            x = x.transpose(1, 2)  # [B, D, T]
            x = self.pointwise_conv1(x)
            x = self.glu(x)
            x = self.depthwise_conv(x)
            x = self.batch_norm(x)
            x = self.activation(x)
            x = self.pointwise_conv2(x)
            x = self.dropout(x)
            x = x.transpose(1, 2)  # [B, T, D]
            return residual + x

    class FeedForwardMacaron(nn.Module):
        """Feed-Forward Network estilo Macaron con factor de escala 0.5."""
        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.layer_norm = nn.LayerNorm(d_model)
            self.linear1 = nn.Linear(d_model, d_ff)
            self.activation = nn.SiLU()
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ff, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.layer_norm(x)
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return residual + 0.5 * self.dropout(x)

    class BloqueConformer(nn.Module):
        """
        Bloque Conformer individual:
        0.5 * FFN -> MHSA -> ConvModule -> 0.5 * FFN -> LayerNorm
        """
        def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 256, conv_kernel: int = 31, dropout: float = 0.1):
            super().__init__()
            self.ffn1 = FeedForwardMacaron(d_model, d_ff, dropout)
            self.self_attn_norm = nn.LayerNorm(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.conv_module = ModuloConvolucionConformer(d_model, conv_kernel, dropout)
            self.ffn2 = FeedForwardMacaron(d_model, d_ff, dropout)
            self.final_norm = nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
            x = self.ffn1(x)
            norm_x = self.self_attn_norm(x)
            attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, key_padding_mask=mask)
            x = x + attn_out
            x = self.conv_module(x)
            x = self.ffn2(x)
            return self.final_norm(x)

    class FusedCrossAttention(nn.Module):
        """
        Módulo de Fusión por Atención Cruzada:
        Las manos (Q) interrogan al contexto multi-stream de Rostro y Torso (K, V).
        """
        def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)
            self.final_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, h_hands: torch.Tensor, h_context: torch.Tensor) -> torch.Tensor:
            q = self.norm_q(h_hands)
            kv = self.norm_kv(h_context)
            out, _ = self.cross_attn(q, kv, kv)
            return self.final_norm(h_hands + self.dropout(out))

    class ConformerMultiStreamLSC(nn.Module):
        def __init__(
            self,
            dim_hands: int = 126,    # 2 manos * 21 pts * 3D
            dim_face: int = 192,     # 64 pts gramaticales * 3D
            dim_pose: int = 99,      # 33 pts de pose * 3D
            d_model: int = 128,      # Dimensión compacta móvil (<12MB)
            n_clases: int = 14,      # 13 clases + blanco CTC
            n_conformer_layers: int = 3,
            n_heads: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()
            self.d_model = d_model
            self.n_clases = n_clases

            # 1. Proyecciones Iniciales de los 3 Streams
            self.proj_hands = nn.Sequential(
                nn.Linear(dim_hands, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU()
            )

            self.proj_face = nn.Sequential(
                nn.Linear(dim_face, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, groups=d_model)
            )

            self.proj_pose = nn.Sequential(
                nn.Linear(dim_pose, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
            )

            # 2. Codificador Dedicado de Manos (Conformer)
            self.conformer_hands = nn.ModuleList([
                BloqueConformer(d_model, n_heads, d_ff=d_model * 2, conv_kernel=31, dropout=dropout)
                for _ in range(2)
            ])

            # 3. Fusión por Atención Cruzada (Manos -> Rostro+Pose)
            self.cross_attention_fusion = FusedCrossAttention(d_model, n_heads, dropout)

            # 4. Bloques Conformer Post-Fusión
            self.conformer_fused = nn.ModuleList([
                BloqueConformer(d_model, n_heads, d_ff=d_model * 2, conv_kernel=31, dropout=dropout)
                for _ in range(n_conformer_layers)
            ])

            # 5. Cabezales de Salida Dual
            # Cabezal CTC para decodificación continua de frases en el tiempo
            self.ctc_head = nn.Linear(d_model, n_clases + 1)  # +1 para token [BLANK] CTC

            # Cabezal de Pooling para clasificación de segmento aislado
            self.pooling_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_clases)
            )

        def forward(
            self,
            x_hands: torch.Tensor,
            x_face: torch.Tensor,
            x_pose: torch.Tensor,
            mask: torch.Tensor = None
        ):
            # Formas de entrada: [B, T, D_in]
            B, T, _ = x_hands.shape

            # 1. Proyección de streams
            h_h = self.proj_hands(x_hands)

            # Para rostro y pose, aplicar conv temporal 1D
            h_f = self.proj_face[:3](x_face).transpose(1, 2)
            h_f = self.proj_face[3](h_f).transpose(1, 2)

            h_p = self.proj_pose[:3](x_pose).transpose(1, 2)
            h_p = self.proj_pose[3](h_p).transpose(1, 2)

            # 2. Conformer sobre stream de manos
            for block in self.conformer_hands:
                h_h = block(h_h, mask)

            # 3. Contexto unificado de Rostro + Torso
            h_context = (h_f + h_p) * 0.5

            # 4. Fusión por Atención Cruzada (Cross-Attention)
            h_fused = self.cross_attention_fusion(h_h, h_context)

            # 5. Conformer profundo post-fusión
            for block in self.conformer_fused:
                h_fused = block(h_fused, mask)

            # 6. Salidas
            # Logits CTC continuos por cada fotograma: [B, T, N_clases + 1]
            ctc_logits = self.ctc_head(h_fused)

            # Pooling temporal medio para seña aislada: [B, N_clases]
            pooled = h_fused.mean(dim=1)
            segment_logits = self.pooling_head(pooled)

            return {
                "ctc_logits": ctc_logits,
                "segment_logits": segment_logits,
                "features": h_fused
            }

else:
    class ConformerMultiStreamLSC:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch es requerido para instanciar ConformerMultiStreamLSC.")
