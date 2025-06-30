#' Raw Data for Dynthetic Data Generation
#'
#' A dataset containing L, R, mean and covariance of G scores, and residuals, all estimated from the real HYPNOS ABPM data
#'
#' @format Five subsets of data, each containing:
#' \describe{
#'   \item{L_tilde}{L matrix after identifiability correction}
#'   \item{R_tilde}{R matrix after identifiability correction}
#'   \item{mean_G}{Mean of G scores}  
#'   \item{cov_G}{Covariance matrices of G scores}
#'   \item{E}{Residuals}
#' }
"synthetic_raw"
