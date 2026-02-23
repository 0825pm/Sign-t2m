import torch
torch.backends.cudnn.enabled = False
import os, sys, numpy as np
import torch.nn.functional as F
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from src.models.nets.sign_denoiser import SignDenoiser
from src.models.nets.text_encoder import CLIP
from diffusers import DDPMScheduler, UniPCMultistepScheduler

DEVICE = 'cuda:0'
DATA = '/home/user/Projects/research/SOKE/data/How2Sign_133d'
NFEATS = 133
B = 32
STEPS = 10000
LR = 1e-3
TEXT_DROP = 0.5

SPINE = [(0,1),(1,2),(2,3),(3,4)]
BODY_C = [(4,7),(4,5),(4,6),(5,8),(6,9),(8,10),(9,11),(10,12),(11,13)]
def _h(w,o):
    c=[]
    for f in range(5):
        b=o+f*3; c+=[(w,b),(b,b+1),(b+1,b+2)]
    return c
ALL_CONN = SPINE + BODY_C + _h(12,14) + _h(13,29)

def to_joints_133(raw):
    T=raw.shape[0]; p=np.zeros((T,1,3),dtype=np.float32)
    b=raw[:,4:43].reshape(T,13,3); b14=np.concatenate([p,b],1)
    lw,rw=b14[:,12:13,:],b14[:,13:14,:]
    lh=raw[:,43:88].reshape(T,15,3)+lw; rh=raw[:,88:133].reshape(T,15,3)+rw
    return np.concatenate([b14,lh,rh],1)

def save_video(gt, gen, path, fps=25):
    T=min(gt.shape[0],gen.shape[0]); fig,(al,ar)=plt.subplots(1,2,figsize=(12,6))
    al.set_title('GT',fontsize=14,fontweight='bold',color='blue')
    ar.set_title('Generated',fontsize=14,fontweight='bold',color='red')
    def setup(ax,j):
        c=j[:T]-j[:T,3:4,:]; x,y=c[:,:,0],-c[:,:,1]
        vp=max(max(x.max()-x.min(),y.max()-y.min())/2,0.15)
        cx,cy=(x.min()+x.max())/2,(y.min()+y.max())/2
        ax.set_xlim(cx-vp,cx+vp);ax.set_ylim(cy-vp,cy+vp);ax.set_aspect('equal');ax.axis('off')
        lines=[]
        for(i,j)in ALL_CONN:
            col=('#E91E63'if(i<29 and j<29)else'#4CAF50')if(i>=14 or j>=14)else('purple'if(i,j)in SPINE else'#333')
            lw=0.8 if(i>=14 or j>=14)else(2.5 if(i,j)in SPINE else 2.0)
            l,=ax.plot([],[],color=col,linewidth=lw,alpha=0.8);lines.append((l,i,j))
        return c,lines,ax.scatter([],[],c='#333',s=15,zorder=5),ax.scatter([],[],c='#E91E63',s=5,zorder=5),ax.scatter([],[],c='#4CAF50',s=5,zorder=5)
    gc,*el=setup(al,gt);gnc,*er=setup(ar,gen)
    ft=fig.text(0.5,0.02,'',ha='center',fontsize=9,color='gray');plt.tight_layout(rect=[0,0.04,1,0.95])
    def upd(f):
        ft.set_text(f'Frame {f}/{T-1}')
        for c,(ls,bs,lhs,rhs)in[(gc,el),(gnc,er)]:
            x,y=c[f,:,0],-c[f,:,1]
            for(l,i,j)in ls:l.set_data([x[i],x[j]],[y[i],y[j]])
            bs.set_offsets(np.c_[x[:14],y[:14]]);lhs.set_offsets(np.c_[x[14:29],y[14:29]]);rhs.set_offsets(np.c_[x[29:44],y[29:44]])
    anim=FuncAnimation(fig,upd,frames=T,interval=1000/fps,blit=False)
    anim.save(path,writer=FFMpegWriter(fps=fps,bitrate=5000));plt.close(fig);print(f"  Saved: {path}")

def main():
    device=torch.device(DEVICE); os.makedirs('overfit_vis_output',exist_ok=True)

    # ─── Load 1 sample (133D npy) ───
    npy_dir=f'{DATA}/train/poses'
    npy_file=sorted(os.listdir(npy_dir))[0]
    raw=np.load(os.path.join(npy_dir,npy_file))
    mean=torch.load(f'{DATA}/mean_133.pt',map_location='cpu').numpy()
    std=torch.load(f'{DATA}/std_133.pt',map_location='cpu').numpy()
    norm=(raw-mean)/(std+1e-10)
    T=(min(norm.shape[0],100)//4)*4
    motion_1=torch.from_numpy(norm[:T]).float().unsqueeze(0).to(device)
    motion=motion_1.repeat(B,1,1)
    print(f"Sample: {npy_file}, T={T}, dim={NFEATS}, B={B}")

    # ─── Build ───
    text_encoder=CLIP().to(device).eval()
    for p in text_encoder.parameters():p.requires_grad_(False)
    denoiser=SignDenoiser(motion_dim=NFEATS,max_motion_len=401,text_dim=512,stage_dim="256*4").to(device).train()
    noise_sched=DDPMScheduler(num_train_timesteps=1000,beta_start=0.0001,beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",clip_sample=False,prediction_type="sample")
    sample_sched=UniPCMultistepScheduler(num_train_timesteps=1000,beta_start=0.0001,beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",prediction_type="epsilon")

    print(f"Params: {sum(p.numel() for p in denoiser.parameters() if p.requires_grad)/1e6:.2f}M")
    optimizer=torch.optim.AdamW(denoiser.parameters(),lr=LR)

    with torch.no_grad():
        # Encode cond+uncond pairs together for matching seq len
        all_texts=["a person signing",""]*B
        all_emb=text_encoder(all_texts,device)
        text_cond={k:v[0::2] for k,v in all_emb.items()}
        text_uncond={k:v[1::2] for k,v in all_emb.items()}
    mask=torch.ones(B,T,dtype=torch.bool,device=device)

    # ═══ PHASE 1: Overfit with text dropout ═══
    print(f"\n=== Overfit (sample pred, text_drop={TEXT_DROP}, {STEPS} steps) ===\n")
    for step in range(STEPS):
        optimizer.zero_grad()
        t=torch.randint(0,1000,(B,),device=device)
        noise=torch.randn_like(motion)
        x_t=noise_sched.add_noise(motion,noise,t)
        drop_mask=torch.rand(B)<TEXT_DROP
        text_input={}
        for k in text_cond:
            text_input[k]=text_cond[k].clone()
            text_input[k][drop_mask]=text_uncond[k][drop_mask]
        pred_x0=denoiser(x_t,mask,t,text_input)
        loss=F.mse_loss(pred_x0,motion)
        loss.backward();optimizer.step()
        if step%1000==0 or step==STEPS-1:
            print(f"  step {step:5d}  loss={loss.item():.6f}")

    # ═══ PHASE 2: Generate with CFG ═══
    print(f"\n=== Generation (CFG=4.0, 50 steps) ===\n")
    denoiser.eval()
    mask_1=torch.ones(1,T,dtype=torch.bool,device=device)
    with torch.no_grad():
        text_cfg=text_encoder(["a person signing",""],device)
        pred_motion=torch.randn(1,T,NFEATS,device=device)
        sample_sched.set_timesteps(50)
        for t_step in sample_sched.timesteps.to(device):
            model_in=pred_motion.repeat(2,1,1)
            mask_2=mask_1.repeat(2,1)
            t_2=t_step.repeat(2)
            pred_x0=denoiser(model_in,mask_2,t_2,text_cfg)
            cond_x0,uncond_x0=pred_x0.chunk(2)
            alpha=noise_sched.alphas_cumprod[t_step.long()]
            cond_eps=(pred_motion-alpha**0.5*cond_x0)/(1-alpha)**0.5
            uncond_eps=(pred_motion-alpha**0.5*uncond_x0)/(1-alpha)**0.5
            guided_eps=uncond_eps+4.0*(cond_eps-uncond_eps)
            pred_motion=sample_sched.step(guided_eps,t_step,pred_motion).prev_sample.float()

    # ═══ PHASE 3: Visualize ═══
    gen_np=pred_motion[0].cpu().numpy(); gt_np=motion_1[0].cpu().numpy()
    gen_raw=gen_np*(std+1e-10)+mean; gt_raw=gt_np*(std+1e-10)+mean
    gen_j=to_joints_133(gen_raw); gt_j=to_joints_133(gt_raw)
    rmse=np.sqrt(((gen_j-gt_j)**2).mean())
    print(f"  RMSE: {rmse:.4f}")
    save_video(gt_j,gen_j,'overfit_vis_output/overfit_133d_cfg.mp4')
    print(f"\n  Done! RMSE={rmse:.4f}")

if __name__=="__main__":
    main()
