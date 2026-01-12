\name{sim_data3}
\alias{sim_data3}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Generation of simulated data
}
\description{
Generate 3-way tensor data when the following information is available. Can define dimension of each mode, noise, noise level, missing pattern, and missing rate.  
}
\usage{
sim_data3(L, b, r2, p, noise_sd, noise_level, pattern, percent, lower, upper)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
 \item{L}{
  A known matrix L of dimension a*r1. Can contain special features such as smoothness.
}
  \item{b}{
  Dimension of the 2nd mode of the sythetic tensor. For example, 3 for DBP, SBP, and HR in ABPM data. Should be an integer.
}
  \item{r2}{
  True decomposition rank for the 2nd mode.
}
  \item{p}{
  Dimension of the 3rd mode of the synthetic tensor. Should be an integer. 
}
  \item{noise_sd}{
  sd value to be used when sampling noise from rnorm with mean=0, sd=noise
}
  \item{noise_level}{
  Noise level. Adjusts noise sampling to be from rnorm with mean=0, sd=noise * sqrt(noise_level). Should be greater than or equal to 0.
}
  \item{pattern}{
  Missing pattern. Can be either "random" or "structured".
}
  \item{percent}{
  Percent of data to be set to missing, if pattern = "random". Should be a number in [0,1).
}
 \item{lower}{
  Minimum number of rows to be masked, if pattern = "structured".
}
  \item{upper}{
  Maximum number of rows to be masked, if pattern = "structured".
}
}
\details{
This function is developed based on the rTensor package (Li et al., 2018). 
}
\value{
\item{sim_Msmooth}{Simulated ground-truth tensor, smooth and complete}
\item{sim_Mmiss}{Simulated tensor, noisy and incomplete}
\item{sim_R}{Simulated R matrix}
\item{sim_G}{Simulated core tensor G}

}
\references{
Li J, Bien J, Wells MT (2018).
\emph{rTensor: An R Package for Multidimensional Array (Tensor) Unfolding, Multiplication, and Decomposition}.
\emph{Journal of Statistical Software}, \bold{87}(10), 1--31. \doi{10.18637/jss.v087.i10}
}
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
  set.seed(1234321)
  
  # Generate random L
  L <- matrix(rnorm(24 * 3), nrow = 24, ncol = 3) # L is 24 * 3
  
  synthetic <- sim_data3(L, b=50, r2=20, p=200, noise_sd=0.6, noise_level=1, pattern="random", percent=0.2)
  
  (synthetic$sim_Msmooth)@data[ , , 1] # ground truth smooth data, showing the first frontal slice
  (synthetic$sim_Mmiss)@data[ , ,1] # noisy, incomplete data, showing the first frontal slice
  synthetic$sim_R
  synthetic$sim_G[ , ,1]
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
