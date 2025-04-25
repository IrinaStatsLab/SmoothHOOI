SecDiffMat <- function(dim){
  D2 <- diag(2,dim,dim)
  D2[row(D2) == col(D2) - 1] <- -1
  D2[row(D2) == col(D2) + 1] <- -1
  D2[dim,1] <- -1
  D2[1,dim] <- -1
  return(D2)
}

# tnsr - tnsr to be masked
# percent - percent of data to be masked
# mask a certain percent of data randomly across the tensor
mask1<-function(tnsr,percent){
  idx <- which(!is.na(tnsr@data))
  rand_sample <- sample(length(idx),ceiling(length(idx)*percent))
  masked_idx <- idx[rand_sample]
  masked_tnsr <- tnsr@data
  masked_tnsr[masked_idx] <- NA
  masked_tnsr <- rTensor::as.tensor(masked_tnsr)
  return(masked_tnsr)
}

# tnsr - tnsr to be masked
# modes - the minimum unit you want to create missing values in
# percent - percent of data to be masked
# mask a certain percent of data for each slice (usually each individual) evenly
mask2<-function(tnsr, modes, percent){ 
  unfolded <- rTensor::unfold(tnsr, row_idx=modes, col_idx = setdiff(seq(1,tnsr@num_modes,by=1),modes))
  n <- dim(unfolded)[1]
  m <- dim(unfolded)[2]
  for (i in 1:m){
    idx <- which(!is.na(unfolded@data[,i]))
    rand_sample <- sample(length(idx),ceiling(length(idx)*percent))
    unfolded[rand_sample,i] <- NA 
  }
  masked_tnsr <- rTensor::fold(unfolded, row_idx = modes, col_idx = setdiff(seq(1,tnsr@num_modes,by=1),modes), modes = tnsr@modes)
  return(masked_tnsr)
}

# tnsr - tnsr to be masked
# modes - the minimum unit you want to create missing values in
# percent - percent of data to be masked
# mask a random percent of data for different slices of the tensor (usually individuals)
mask3<-function(tnsr, modes){ 
  unfolded <- rTensor::unfold(tnsr, row_idx=modes, col_idx = setdiff(seq(1,tnsr@num_modes,by=1),modes))
  n <- dim(unfolded)[1]
  m <- dim(unfolded)[2]
  percent <- c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
  for (i in 1:m){
    idx <- which(!is.na(unfolded@data[,i]))
    percent_i <- sample(percent,1)
    rand_sample <- sample(length(idx),ceiling(length(idx)*percent_i))
    unfolded[rand_sample,i] <- NA 
  }
  masked_tnsr <- rTensor::fold(unfolded, row_idx = modes, col_idx = setdiff(seq(1,tnsr@num_modes,by=1),modes), modes = tnsr@modes)
  return(masked_tnsr)
}

# tnsr - tnsr to be masked 
# this function is to mask the data in  a structured way. 
# E.g. for each patient, DBP/SBP/HR are missing at the same time, and each patient can have 0-20 timepoints missing
mask4 <- function(tnsr){
  n <- tnsr@modes[3]
  for (i in 1:n){
    mask_num <- sample(c(0:20), 1)
    mask_tp <- sample(c(1:24), mask_num)
    tnsr@data[mask_tp, ,i] <- NA
  }
  return(tnsr)
}

simdata_generator <- function(L, G, R, E, p, noise_level, pattern, percent){
  G_mat <- cbind(G[1,1,], G[1,2,], 
                 G[2,1,], G[2,2,],
                 G[3,1,], G[3,2,])
  
  mean_G <- colMeans(G_mat)
  cov_G <- cov(G_mat)
  
  error_sd <- sd(E, na.rm=T)
  
  sim_G <- array(NA, dim=c(3, 2, p))
  for (i in 1:p){
    sim <- MASS::mvrnorm(n = 1, mu = mean_G, Sigma = cov_G)
    sim_G[,,i] <- matrix(sim, nrow=3, byrow=TRUE)
  }
  
  sim_Msmooth <- rTensor::ttl(rTensor::as.tensor(sim_G), list_mat=list(L, R), ms=c(1,2))
  
  error_array <- array(rnorm(24*3*p, mean=0, sd=error_sd*noise_level), dim=c(24,3,p))
  
  sim_Mnoise <- rTensor::as.tensor(sim_Msmooth@data + error_array)
  
  if (pattern=="random"){
    sim_Mmiss <- mask1(sim_Mnoise, percent = percent)
  } else if (pattern=="structured"){
    sim_Mmiss <- mask4(sim_Mnoise)
  }
  
  out=list(sim_Msmooth=sim_Msmooth, sim_Mmiss=sim_Mmiss)
}

fpca_res <- function(tnsr, smooth_tnsr, true_L, npc=NULL, pve=0.99, center=TRUE){
  nmiss_idx <- which(!is.na(tnsr@data))
  
  h1 <- rTensor::unfold(tnsr[,1,], row_idx = 2, col_idx = 1)@data
  h2 <- rTensor::unfold(tnsr[,2,], row_idx = 2, col_idx = 1)@data
  h3 <- rTensor::unfold(tnsr[,3,], row_idx = 2, col_idx = 1)@data
  
  outfpca1 <- refund::fpca.sc(Y = as.matrix(h1), pve=pve, npc = npc, center=center)
  outfpca2 <- refund::fpca.sc(Y = as.matrix(h2), pve=pve, npc = npc, center=center)
  outfpca3<- refund::fpca.sc(Y = as.matrix(h3), pve=pve, npc = npc, center=center)
  
  p <- tnsr@modes[3]
  outfpcaYhat <- array(NA, dim=c(24,3,p))
  for (i in 1:p){
    outfpcaYhat[,1,i] <- outfpca1$Yhat[i,]
    outfpcaYhat[,2,i] <- outfpca2$Yhat[i,]
    outfpcaYhat[,3,i] <- outfpca3$Yhat[i,]
  }
  
  loss_M <- sum((outfpcaYhat - smooth_tnsr@data)^2)/length(outfpcaYhat)
  
  min_col <- min(ncol(outfpca1$efunctions), ncol(outfpca2$efunctions), ncol(outfpca3$efunctions))
  
  Lfpca1 <- outfpca1$efunctions[,1:min_col]
  Lfpca2 <- outfpca2$efunctions[,1:min_col]
  Lfpca3 <- outfpca3$efunctions[,1:min_col]
  
  Lfpca1 <- qr.Q(qr(Lfpca1))
  Lfpca2 <- qr.Q(qr(Lfpca2))
  Lfpca3 <- qr.Q(qr(Lfpca3))
  
  Lfpca <- (Lfpca1 + Lfpca2 + Lfpca3)/3
  Lfpca <- qr.Q(qr(Lfpca))
  
  loss_L <- sqrt(0.5*sum((true_L %*% t(true_L) - Lfpca %*% t(Lfpca))^2))
  
  out=list(est=outfpcaYhat, Lfpca1=Lfpca1, Lfpca2=Lfpca2, Lfpca3=Lfpca3, Lfpca=Lfpca,
    loss_M=loss_M, loss_L=loss_L)
}


# Rotation to guarantee identifiability for downstream inference
# L - L generated by cglram/mglram
# G - G generated by cglram/mglram
# R - R generated by cglram/mglram
MakeIdent <- function(L, G, R){
  unfold1 <- rTensor::k_unfold(rTensor::as.tensor(G), m = 1) # Unfold by mode-1
  svdmode1 <- svd(unfold1@data)
  U_1 = svdmode1$u
  
  unfold2 <- rTensor::k_unfold(rTensor::as.tensor(G), m = 2) # Unfold by mode-2
  svdmode2 <- svd(unfold2@data)
  U_2 = svdmode2$u
  
  n = dim(G)[3] 
  G_tilde <- array(NA, dim = dim(G))
  for (i in 1:n){
    G_tilde[, , i] = t(U_1) %*% G[, , i] %*% U_2 
  }
  
  L_tilde = L %*% U_1 
  R_tilde = R %*% U_2
  
  out = list(L_tilde = L_tilde, R_tilde = R_tilde, G_tilde = G_tilde)
}


