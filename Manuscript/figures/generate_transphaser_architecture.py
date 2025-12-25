
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import numpy as np

def draw_transphaser_architecture():
    # Setup figure with professional B&W style
    # ADJUSTED physical size to be more standard (15x9) so fonts aren't tiny when scaled
    # Aspect ratio matches 20/12 coordinate system approx
    fig = plt.figure(figsize=(15, 9)) 
    ax = fig.add_axes([0, 0, 1, 1]) 
    ax.set_xlim(0, 20) 
    ax.set_ylim(0, 12) # Height 12
    ax.axis('off')
    
    # SHIFT EVERYTHING UP BY DY
    DY = 2.0
    
    # --- Constants & Helpers ---
    LW_BOX = 1.5
    LW_ARROW = 1.2
    FONT = 'serif' 
    
    def add_box(x, y, w, h, label, eq=None, dashed=False, bg_color='white'):
        # APPLY SHIFT
        y = y + DY
        
        ls = '--' if dashed else '-'
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      fc=bg_color, ec="black", lw=LW_BOX, ls=ls)
        ax.add_patch(rect)
        
        # Centered text
        cx, cy = x + w/2, y + h/2
        if label:
            ax.text(cx, cy + (0.2 if eq else 0), label, ha='center', va='center', 
                    fontsize=11, fontweight='bold', fontfamily=FONT)
        if eq:
            ax.text(cx, cy - (0.2 if label else 0), eq, ha='center', va='center', 
                    fontsize=10, fontfamily=FONT, fontstyle='italic')
        return x, y, w, h
    
    def arrow(x1, y1, x2, y2, label=None):
        # APPLY SHIFT
        y1 = y1 + DY
        y2 = y2 + DY
        
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                    arrowprops=dict(arrowstyle="->", color="black", lw=LW_ARROW))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', va='bottom', 
                    fontsize=9, fontfamily=FONT,  bbox=dict(fc='white', ec='none', pad=0.5))

    # ==========================
    # COLUMN 1: INPUTS
    # ==========================
    # Main container implies 'Observed Data'
    
    # Unphased Genotype
    # Manually placing label at top
    # Box is at (0.5, 4.5+DY), height 2.0
    G_BOX_Y = 4.5 + DY
    G_BOX_H = 2.0
    add_box(0.5, 4.5, 2.0, 2.0, "", r"$G = \{a_{i,1}, a_{i,2}\}$")
    
    # Label "Genotype G" at the very top inside the box, bold
    ax.text(1.5, G_BOX_Y + G_BOX_H - 0.25, "Genotype G", ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Visual grid for genotype alleles - moved further down to avoid overlap
    ax.text(1.5, G_BOX_Y + 0.5, "A*01, A*02\nB*35, B*44\n...", ha='center', va='center', fontsize=9, fontfamily='monospace', color='#444')

    # ==========================
    # COLUMN 2: NEURAL PROPOSAL (Amortized)
    # ==========================
    # Arrow G -> Net
    arrow(2.6, 5.5, 3.5, 5.5)
    
    # Big Box: Proposal Network
    add_box(3.5, 1.0, 3.5, 6.0, "", dashed=True, bg_color='#f9f9f9')
    ax.text(5.25, 6.5 + DY, "Neural Proposal Network\n(Amortized Inference)", ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(5.25, 0.6 + DY, r"$q_\phi(H|G,x)$", ha='center', fontsize=12)
    
    # Internal components
    # 1. Embedding
    add_box(3.8, 5.2, 2.9, 0.8, "Allele Embeddings")
    arrow(5.25, 5.2, 5.25, 4.8)
    
    # 2. Transformer
    add_box(3.8, 2.8, 2.9, 2.0, "Transformer Encoder", r"Self-Attention(LD)")
    arrow(5.25, 2.8, 5.25, 2.4)
    
    # 3. Head
    add_box(3.8, 1.4, 2.9, 1.0, "Phasing Head", r"$P(phase_i|ctx)$")

    # ==========================
    # COLUMN 3: CANDIDATES (Reified Latents)
    # ==========================
    arrow(7.0, 4.0, 8.0, 4.0, "Propose Top-K")
    
    # Container
    add_box(8.0, 1.0, 3.0, 6.0, "", bg_color='#fff')
    ax.text(9.5, 6.7 + DY, "Candidate Haplotypes", ha='center', fontsize=12, fontweight='bold')
    
    # Stack of pairs
    for i in range(3):
        y = 5.2 - i*1.5
        add_box(8.2, y, 2.6, 1.0, f"Candidate {i+1}", r"$(H_1^{(k)}, H_2^{(k)})$", dashed=True)
    
    # ==========================
    # COLUMN 4: PROBABILISTIC MODEL (Scoring)
    # ==========================
    arrow(11.0, 4.0, 12.0, 4.0)
    
    # Container
    add_box(12.0, 1.0, 4.0, 6.0, "", bg_color='#f9f9f9')
    ax.text(14.0, 6.7 + DY, "Probabilistic Scoring Model", ha='center', fontsize=12, fontweight='bold')
    
    # 1. Conditional Prior (Neural)
    add_box(12.3, 5.0, 3.4, 1.2, "Conditional Prior", r"$\pi_\theta(h|x) = \text{softmax}(f_\theta(h,x))$")
    arrow(14.0, 5.0, 14.0, 4.6)
    
    # 2. HWE (Structural)
    add_box(12.3, 3.2, 3.4, 1.2, "HWE Pair Prior", r"$P(H_1, H_2) \propto \pi(H_1)\pi(H_2)$")
    arrow(14.0, 3.2, 14.0, 2.8)
    
    # 3. Emission (Constraint)
    add_box(12.3, 1.4, 3.4, 1.2, "Emission Constraint", r"$P(G|H) = \mathbb{I}(\{H_1, H_2\} \equiv G)$")
    
    # G Consistency check path
    # Manually shift coordinates
    # path was: ([1.5, 1.5, 11.6, 11.6, 12.3], [4.5, 0.4, 0.4, 2.0, 2.0])
    # shift y by DY
    path_x = [1.5, 1.5, 11.6, 11.6, 12.3]
    path_y = [4.5+DY, 0.4+DY, 0.4+DY, 2.0+DY, 2.0+DY]
    
    ax.plot(path_x, path_y, color='black', lw=1, ls=':')
    ax.text(7.0, 0.5 + DY, "Genotype Consistency Check", fontsize=9, bbox=dict(fc='white', ec='none'))

    # ==========================
    # COLUMN 5: OBJECTIVE
    # ==========================
    # Objective at top
    # SHIFTING UP: y=7.2 -> 8.5 (+1.3 shift)
    # y=7.2+DY=9.2 -> New y=10.5
    # Box: (12.0, 8.5+DY)
    OBJ_Y = 8.5 # Base Y coord for objective
    
    add_box(12.0, OBJ_Y, 5.0, 0.8, "Training Objective", r"$\max \sum_k w_k \log P(H^{(k)}, G) - \beta \text{KL}$")
    
    # Arrows adjusted to reach new height
    # Model (x=14.0) connects up to Objective
    # Arrow 1: From Model Top (Prior Box top ~ y=6.2+DY) to Objective Bottom
    arrow(14.0, 7.0, 14.0, OBJ_Y) 
    
    # Posterior Probs output
    # Arrow coords: (16.0, 4.0) -> (17.0, 4.0) shifted by DY
    arrow(16.0, 4.0, 17.0, 4.0)
    # Box moved RIGHT to x=17.2 from 16.5 so arrow tip (x=17.0) touches edge, not inside
    add_box(17.2, 2.5, 1.4, 3.0, "Post.\nProbs\n$r_k$")
    
    plt.tight_layout()
    plt.savefig('Manuscript/figures/transphaser_architecture.pdf', bbox_inches='tight')
    plt.savefig('Manuscript/figures/transphaser_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('Manuscript/figures/transphaser_architecture.svg', bbox_inches='tight')

if __name__ == "__main__":
    draw_transphaser_architecture()
