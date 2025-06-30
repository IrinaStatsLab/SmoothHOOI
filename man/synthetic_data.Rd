\name{synthetic_data}
\alias{synthetic_data}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Generation of simulated data
}
\description{
Generate simulated data when the following data are available. Can also define sample size (p), noise level, missing pattern, and missing rate.  
}
\usage{
synthetic_data(L, R, mean_G, cov_G, E, p, noise_level, pattern, percent)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{L}{
  A known matrix L of dimension a*r1. 
}
  \item{R}{
  A known matrix R of dimension b*r2. 
}
  \item{mean_G}{
  A known vector of mean values for G scores.
}
  \item{cov_G}{
  A known matrix for the covariance of G scores.  
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
%\examples{

%}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
