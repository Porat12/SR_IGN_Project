import torch
import torch.nn.functional as F

from .loss_registry import register_loss


@register_loss("SR_IGN_loss_for_train")
def SR_IGN_loss_for_train(f, f_copy, z, x, **loss_params):

    lam_rec, lam_idem, lam_tight  = loss_params["lam_rec"], loss_params["lam_idem"], loss_params["lam_tight"]
    lam_SR, a = loss_params["lam_SR"], loss_params.get("a", None)

    fx = f(x)
    fz = f(z)
    f_z = fz.detach()
    ff_z = f(f_z)
    f_fz = f_copy(fz)

    loss_SR = F.mse_loss(fz, x)

    loss_rec = F.mse_loss(fx, x)
    detached_loss_rec = loss_rec.detach()

    loss_idem = F.mse_loss(f_fz, fz)

    loss_tight = -F.mse_loss(ff_z, f_z)
    if a is not None:
        loss_tight = F.tanh( loss_tight / (a * detached_loss_rec) ) * detached_loss_rec

    loss = lam_rec * loss_rec + lam_idem * loss_idem + lam_tight * loss_tight + lam_SR * loss_SR

    info = {"rec_loss": loss_rec.item(),
            "idem_loss": loss_idem.item(),
            "tight_loss": loss_tight.item(),
            "SR_loss": loss_SR.item()}
    
    return loss, info


@register_loss("SR_IGN_loss_for_test")
@torch.no_grad()
def SR_IGN_loss_for_test(f, z, x, **loss_params):

    lam_rec, lam_idem, lam_tight  = loss_params["lam_rec"], loss_params["lam_idem"], loss_params["lam_tight"]
    lam_SR, a = loss_params["lam_SR"], loss_params.get("a", None)

    fx = f(x)
    fz = f(z)
    ffz= f(fz)

    loss_SR = F.mse_loss(fz, x)

    loss_rec = F.mse_loss(fx, x)

    loss_idem = F.mse_loss(ffz, fz)

    loss_tight = -loss_idem
    if a is not None:
        loss_tight = F.tanh( loss_tight / (a * loss_rec) ) * loss_rec

    loss = lam_rec * loss_rec + lam_idem * loss_idem + lam_tight * loss_tight + lam_SR * loss_SR

    info = {"rec_loss": loss_rec.item(),
            "idem_loss": loss_idem.item(),
            "tight_loss": loss_tight.item(),
            "SR_loss": loss_SR.item()}
    
    return loss, info

@register_loss("SR_loss_for_train")
def SR_loss_for_train(f, f_copy, z, x, **loss_params):


    fz = f(z)

    loss_SR = F.mse_loss(fz, x)

    loss = loss_SR

    info = {"rec_loss": 0.0,
            "idem_loss": 0.0,
            "tight_loss": 0.0,
            "SR_loss": loss_SR.item()}
    
    return loss, info

@register_loss("SR_loss_for_test")
@torch.no_grad()
def SR_loss_for_test(f, z, x, **loss_params):

    fz = f(z)

    loss_SR = F.mse_loss(fz, x)


    loss = loss_SR

    info = {"rec_loss": 0.0,
            "idem_loss": 0.0,
            "tight_loss": 0.0,
            "SR_loss": loss_SR.item()}
    
    return loss, info