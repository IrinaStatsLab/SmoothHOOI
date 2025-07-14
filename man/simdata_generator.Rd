\name{simdata_generator}
\alias{simdata_generator}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Generation of simulated data 
}
\description{
Generate simulated ABPM data of 24(hours)*3(DBP, SBP, HR)*p(sample size) when the following information is available. Can define sample size (p), noise level, missing pattern, and missing rate.  
}
\usage{
simdata_generator(L, G, R, E, p, noise_level, pattern, percent)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{L}{
  A known matrix L of dimension a*r1. 
}
  \item{G}{
  A known array G of dimension r1*r2*n. 
}
  \item{R}{
  A known matrix R of dimension b*r2. 
}
  \item{E}{
  A known array E of dimension a*b*n, containing the empirical errors.
}
  \item{p}{
  Sample size. Should be an integer. 
}
  \item{noise_level}{
  Noise level. Should be greater than or equal to 0.
}
  \item{pattern}{
  Missing pattern. Can be either "random" or "structured".
}
  \item{percent}{
  Percent of data to be set to missing, if pattern = "random". Should be a number in [0,1).
}
}
\details{
This function is developed based on the rTensor package (Li et al., 2018). 
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{sim_Msmooth}{Simulated ground-truth tensor, smooth and complete}
\item{sim_Mmiss}{Simulated tensor, noisy and incomplete}
%% ...
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
  
  # Generate random L, G, R, E (does not have specific meaning, just for example) 
  L <- matrix(rnorm(24 * 3), nrow = 24, ncol = 3) # L is 24 * 3
  G <- array(rnorm(3 * 2 * 207), dim = c(3, 2, 207)) # G is 24 * 3 * 207
  R <- matrix(rnorm(3 * 2), nrow = 3, ncol = 2) # R is 3 * 2
  E <- array(rnorm(24 * 3 * 207), dim = c(24, 3, 207)) # E is 24 * 3 * 207
  
  synthetic <- simdata_generator(L, G, R, E, p = 207, noise_level = 1, pattern="random", percent = 0.2)
  
  (synthetic$sim_Msmooth)@data[ , , 1] # ground truth smooth data, showing the first frontal slice
  (synthetic$sim_Mmiss)@data[ , ,1] # noisy, incomplete data, showing the first frontal slice
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
