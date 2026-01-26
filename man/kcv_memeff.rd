\name{kcv_memeff}
\alias{kcv_memeff}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Selection of optimal hyperparameters with K-fold cross-validation, a more memory-efficient version 
}
\description{
Find the optimal hyperparameters that minimize the cross-validation (CV) error in a certain search grid. The memory usage is improved compared to kcv(), which is more suitable for extremely large tensors.
}
\usage{
kcv_memeff(tnsr, rank_grid, lambda_seq, k, L0, D, tol = 0.1, max_iter = 500L, init = 0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tnsr}{
  A multi-dimensional array of dimension a*b*n. Should be organized so that the first two dimensions are to be compressed.
}
  \item{rank_grid}{
  A matrix in which each row is a combination of ranks r1 and r2. 
}
  \item{lambda_seq}{
  A vector of tuning parameters. 
}
  \item{k}{
  Number of folds.
}
  \item{L0}{
  A matrix of dimension a*r1 that initializes L. Can be set to NULL.
}
  \item{D}{
   A diffenrence matrix of dimension a*a. 
}
  \item{tol}{
  Tolerance that defines convergence criteria.
}
  \item{max_iter}{
  Maximum of number of iterations. 
}
  \item{init}{
  A constant that imputes the missing data for initialization purpose.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{MSE_mat}{A matrix of CV error; rows representing ranks, cols representing lambda}
\item{SE_mat}{A matrix of standard errors of the CV error across the k folds}
\item{opt_para}{A vector of optimal hyperparameters; in the order of r1, r2, lambda}
%% ...
}
%\references{
%% ~put references to the literature/web site here ~
%}
\author{
Leyuan Qian
}
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
  # generate a random array with missing data
  dims <- c(24, 3, 20)
  set.seed(123) 
  tnsr <- array(rnorm(prod(dims), mean = 0, sd = 1), dim = dims)
  
  missing_prob <- 0.2
  missing_indices <- sample(length(tnsr), size = floor(length(tnsr) * missing_prob))
  tnsr[missing_indices] <- NA
  
  # generate a second order difference matrix with circular nature 
  # (can actually use SecDiffMat() function provided by this package)
  D2<-diag(2,24,24)
  D2[row(D2) == col(D2) - 1] <- -1
  D2[row(D2) == col(D2) + 1] <- -1
  D2[24,1] <- -1
  D2[1,24] <- -1
  
  # Run 5-fold cross-validation
  kcv_res <- kcv_memeff(tnsr, rank_grid=as.matrix(expand.grid(r1<-c(3), r2<-c(2))), 
                        lambda_seq=seq(2,3,by=1), 
                        k=5, L0=NULL, D=D2, tol=0.01, max_iter=500, init=0)
  
  kcv_res$MSE_mat # matrix of CV error
  kcv_res$opt_para # optimal hyperparameters
}
