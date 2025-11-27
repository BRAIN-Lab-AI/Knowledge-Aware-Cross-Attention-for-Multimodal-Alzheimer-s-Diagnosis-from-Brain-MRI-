"""
imp_model_pretrain3D.py
🔥 IMPROVED MODEL - Properly extends ALBEF3D architecture
This version correctly inherits from and enhances the original ALBEF3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# Import the original ALBEF3D components
from model_pretrain3D import ALBEF3D, VisionTransformer3D, concat_all_gather

# ============================================================================
# ✨ NEW MODULE: Hierarchical Cross-Modal Fusion (Solution A)
# ============================================================================
class HierarchicalCrossModalFusion(nn.Module):
    """
    🔥 IMPROVEMENT: Replaces simple BERT fusion with hierarchical attention
    
    Stage 1: Cross-attention between MRI and PET patch embeddings
    Stage 2: Self-attention on fused features
    Stage 3: Projection to final fused representation
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Stage 1: Cross-modal attention
        self.mri_to_pet_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pet_to_mri_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Stage 2: Self-attention for refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Stage 3: Final projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, mri_embeds, pet_embeds, mri_atts=None, pet_atts=None):
        """
        Args:
            mri_embeds: [batch, num_patches+1, embed_dim]
            pet_embeds: [batch, num_patches+1, embed_dim]
        
        Returns:
            fused_embeds: [batch, num_patches+1, embed_dim] - Fused representations
        """
        batch_size = mri_embeds.size(0)
        
        # Stage 1: Bidirectional cross-attention
        # MRI attends to PET
        mri_enhanced, _ = self.mri_to_pet_attn(
            query=mri_embeds,
            key=pet_embeds,
            value=pet_embeds
        )
        mri_enhanced = self.norm1(mri_embeds + self.dropout(mri_enhanced))
        
        # PET attends to MRI
        pet_enhanced, _ = self.pet_to_mri_attn(
            query=pet_embeds,
            key=mri_embeds,
            value=mri_embeds
        )
        pet_enhanced = self.norm2(pet_embeds + self.dropout(pet_enhanced))
        
        # Stage 2: Concatenate and apply self-attention
        combined = torch.cat([mri_enhanced, pet_enhanced], dim=1)
        
        refined, _ = self.self_attn(
            query=combined,
            key=combined,
            value=combined
        )
        refined = self.norm3(combined + self.dropout(refined))
        
        # Extract and project CLS tokens
        mri_cls = refined[:, 0, :]
        pet_cls = refined[:, mri_embeds.size(1), :]
        
        fused_cls = self.fusion_proj(torch.cat([mri_cls, pet_cls], dim=1))
        
        # Create output with fused CLS and original patch structure
        output_embeds = mri_enhanced.clone()
        output_embeds[:, 0, :] = fused_cls  # Replace CLS token with fused version
        
        return output_embeds


# ============================================================================
# ✨ NEW MODULE: Uncertainty-Aware Classification Head (Solution B)
# ============================================================================
class UncertaintyAwareClassifier(nn.Module):
    """
    🔥 IMPROVEMENT: Adds uncertainty quantification using Monte Carlo Dropout
    """
    def __init__(self, in_features=768, num_classes=3, hidden_size=384, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        self.dropout_p = dropout
    
    def forward(self, x, n_samples=1, return_uncertainty=False):
        """
        Args:
            x: [batch, embed_dim]
            n_samples: Number of MC dropout samples
            return_uncertainty: Whether to return uncertainty scores
        """
        if not return_uncertainty or n_samples <= 1:
            return self.classifier(x)
        
        # Monte Carlo Dropout
        self.train()
        predictions = []
        for _ in range(n_samples):
            predictions.append(self.classifier(x))
        self.eval()
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        
        if return_uncertainty:
            uncertainty = predictions.std(dim=0)
            return mean_pred, uncertainty
        
        return mean_pred


# ============================================================================
# MAIN MODEL: ImprovedALBEF3D (Properly extends ALBEF3D)
# ============================================================================
class ImprovedALBEF3D(ALBEF3D):
    """
    🔥 IMPROVED VERSION - Properly extends ALBEF3D with enhanced fusion
    
    IMPROVEMENTS:
    1. HierarchicalCrossModalFusion for better multimodal integration
    2. UncertaintyAwareClassifier for confidence estimation
    3. Maintains ALL original ALBEF functionality (momentum encoders, ITA, ITM losses)
    """
    def __init__(self, config=None, patch_size=16):
        super().__init__(config, patch_size)
        
        # ✨ NEW: Replace simple BERT fusion with hierarchical fusion
        self.hierarchical_fusion = HierarchicalCrossModalFusion(
            embed_dim=config['vision_width'],
            num_heads=config.get('num_heads', 8),
            dropout=0.1
        )
        
        # ✨ NEW: Uncertainty-aware classification
        self.use_uncertainty = config.get('use_uncertainty', False)

    def forward(self, mri, pet, alpha=0, return_uncertainty=False):
        """
        Enhanced forward with hierarchical fusion
        Maintains compatibility with original training loop
        """
        # ✅ ORIGINAL: Get base features (unchanged)
        image_embeds = self.visual_encoder(mri) 
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(mri.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]), dim=-1)

        pet_embeds = self.pet_encoder(pet)
        pet_atts = torch.ones(pet_embeds.size()[:-1], dtype=torch.long).to(pet.device)
        pet_feat = F.normalize(self.pet_proj(pet_embeds[:,0,:]), dim=-1)

        # 🔥 IMPROVEMENT: Use hierarchical fusion instead of direct BERT fusion
        fused_embeds = self.hierarchical_fusion(
            image_embeds, pet_embeds, image_atts, pet_atts
        )

        # ✅ ORIGINAL: ITA Loss (unchanged, uses original features)
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(mri) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]), dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         
            
            pet_embeds_m = self.pet_encoder_m(pet) 
            pet_feat_m = F.normalize(self.pet_proj_m(pet_embeds_m[:,0,:]), dim=-1) 
            pet_feat_all = torch.cat([pet_feat_m.t(),self.pet_queue.clone().detach()],dim=1)

            sim_i2p_m = image_feat_m @ pet_feat_all / self.temp 
            sim_p2i_m = pet_feat_m @ image_feat_all / self.temp 

            sim_targets = torch.zeros(sim_i2p_m.size()).to(mri.device)
            sim_targets.fill_diagonal_(1)
            
            sim_i2p_targets = alpha * F.softmax(sim_i2p_m, dim=1) + (1 - alpha) * sim_targets
            sim_p2i_targets = alpha * F.softmax(sim_p2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2p = image_feat @ pet_feat_all / self.temp 
        sim_p2i = pet_feat @ image_feat_all / self.temp 
                             
        loss_i2p = -torch.sum(F.log_softmax(sim_i2p, dim=1)*sim_i2p_targets,dim=1).mean()
        loss_p2i = -torch.sum(F.log_softmax(sim_p2i, dim=1)*sim_p2i_targets,dim=1).mean() 

        loss_ita = (loss_i2p + loss_p2i)/2
        self._dequeue_and_enqueue(image_feat_m, pet_feat_m)

        # 🔥 IMPROVEMENT: ITM Loss with hierarchically fused features
        # Use fused embeddings for positive examples
        output_pos = self.fusion_encoder(
            encoder_embeds=pet_embeds, 
            attention_mask=pet_atts,
            encoder_hidden_states=fused_embeds,  # Use fused instead of original image_embeds
            encoder_attention_mask=image_atts,      
            return_dict=True,
            mode='multimodal',
        )
        
        # ✅ ORIGINAL: Negative sampling (unchanged)
        with torch.no_grad():
            bs = mri.size(0)
            weights_i2p = F.softmax(sim_i2p[:,:bs],dim=1)
            weights_p2i = F.softmax(sim_p2i[:,:bs],dim=1)
            weights_i2p.fill_diagonal_(0)
            weights_p2i.fill_diagonal_(0)

        image_embeds_neg = []    
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_p2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   
        image_atts_neg = torch.stack(image_atts_neg,dim=0)      

        pet_embeds_neg = []
        pet_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2p[b], 1).item()
            pet_embeds_neg.append(pet_embeds[neg_idx])
            pet_atts_neg.append(pet_atts[neg_idx])
        pet_embeds_neg = torch.stack(pet_embeds_neg,dim=0)   
        pet_atts_neg = torch.stack(pet_atts_neg,dim=0)

        pet_embeds_all = torch.cat([pet_embeds, pet_embeds_neg],dim=0)
        pet_atts_all = torch.cat([pet_atts, pet_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts],dim=0)

        # Use original BERT fusion for negatives (maintains compatibility)
        output_neg = self.fusion_encoder(
            encoder_embeds=pet_embeds_all, 
            attention_mask=pet_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,      
            return_dict=True,
            mode='multimodal',
        )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]], dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],dim=0).to(mri.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     

        # MLM loss placeholder
        loss_mlm = torch.tensor(0.0).to(mri.device) 
        
        return loss_mlm, loss_ita, loss_itm


# ============================================================================
# IMPROVED CLASSIFIER: Enhanced version of MultiModal3DClassifier
# ============================================================================
class ImprovedMultiModal3DClassifier(ImprovedALBEF3D):
    """
    🔥 IMPROVED CLASSIFIER - Extends ImprovedALBEF3D for classification
    """
    def __init__(self, config=None, patch_size=16, class_num=3):
        super().__init__(config, patch_size)
        
        # ✨ NEW: Uncertainty-aware classification head
        self.cls_head = UncertaintyAwareClassifier(
            in_features=config['vision_width'],
            num_classes=class_num,
            hidden_size=config.get('hidden_size', 384),
            dropout=0.3
        )
        
        self.class_num = class_num
        self.use_uncertainty = config.get('use_uncertainty', False)

    def forward(self, mri, pet, label=None, alpha=0, train=True, return_uncertainty=False):
        """
        Enhanced classification forward with hierarchical fusion and uncertainty
        """
        # Get embeddings from both modalities
        image_embeds = self.visual_encoder(mri) 
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(mri.device)
        
        pet_embeds = self.pet_encoder(pet)
        pet_atts = torch.ones(pet_embeds.size()[:-1], dtype=torch.long).to(pet.device)

        # 🔥 IMPROVEMENT: Use hierarchical fusion
        fused_embeds = self.hierarchical_fusion(
            image_embeds, pet_embeds, image_atts, pet_atts
        )
        
        # Get fused CLS token for classification
        fused_cls = fused_embeds[:, 0, :]

        # 🔥 IMPROVEMENT: Uncertainty-aware classification
        if self.use_uncertainty and not train and return_uncertainty:
            prediction, uncertainty = self.cls_head(
                fused_cls, n_samples=10, return_uncertainty=True
            )
            self.last_uncertainty = uncertainty
        else:
            prediction = self.cls_head(fused_cls)

        # Calculate loss
        loss_cls = torch.tensor(0.0).to(mri.device)
        if label is not None:
            loss_cls = F.cross_entropy(prediction, label)

        # Return compatibility format
        loss_mlm = torch.tensor(0.0).to(mri.device)
        loss_ita = torch.tensor(0.0).to(mri.device)
        loss_itm = torch.tensor(0.0).to(mri.device)

        return loss_cls, loss_mlm, loss_ita, loss_itm, prediction

    def get_metrics(self, output, label):
        """✅ ORIGINAL: Calculate accuracy (unchanged)"""
        preds = torch.argmax(output, dim=1)
        acc = (preds == label).float().mean()
        return {'acc': acc.item()}

    def get_uncertainty(self):
        """✨ NEW: Get uncertainty scores"""
        return getattr(self, 'last_uncertainty', None)