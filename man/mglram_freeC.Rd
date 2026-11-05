\name{mglram_freeC}
\alias{mglram_freeC}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
SmoothHOOI implementation with additional 3rd mode compression
}
\description{
Extend the SmoothHOOI algorithm to compress all three modes, with pre-defined ranks, tuning parameters, and difference matrix
}
\usage{
mglram_freeC(tnsr, ranks, lambda, L0 = NULL, C0 = NULL, D, tol = 1e-05, max_iter = 500, init = 0, identify = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tnsr}{
A multi-dimensional array of dimension a*b*n. 
}
  \item{ranks}{
A vector containing desired ranks r1, r2, and r3.
}
  \item{lambda}{
Tuning parameter.
}
  \item{L0}{
A matrix of dimension a*r1 that initializes L. Can be set to NULL.
}
  \item{C0}{
A matrix of dimension n*r3 that initializes C (the third mode factor matrix). Can be set to NULL.
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
  \item{identify}{
Logical. Whether to perform identifiability correction
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
\item{L}{First-mode factor matrix (a*r1)}
\item{R}{Second-mode factor matrix (b*r2)}
\item{C}{Third-mode factor matrix (n*r3)}
\item{G}{Core tensor (r1*r2*r3)}
\item{est}{Estimated tensor}
\item{obj_vec}{Vector containing objective values through updates}
\item{conv}{A boolean that tells whether the algorithm converges}
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

% \seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
% }
% \examples{

% }
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
