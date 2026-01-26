\name{mglram}
\alias{mglram}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
SmoothHOOI implementation
}
\description{
Solve the SmoothHOOI problem with pre-defined ranks, tuning parameters, and difference matrix.
}
\usage{
mglram(tnsr, ranks, lambda, L0, D, tol = 1e-05, max_iter = 500L, init = 0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tnsr}{
  A multi-dimensional array of dimension a*b*n. Should be organized so that the first two dimensions are to be compressed.
}
  \item{ranks}{
  A vector containing desired ranks r1 and r2.
}
  \item{lambda}{
  Tuning parameter.
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
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{L}{Left orthogonal matrix (a*r1), which contains the components for the 1st mode}
\item{R}{Right orthogonal matrix (b*r2), which contains the components for the 2nd mode}
\item{G}{Core tensor (r1*r2*n), in the array form}
\item{est}{Estimated tensor LGR^T (a*b*n), in the array form}
\item{conv}{A boolean that tells whether the algorithm converges}
\item{obj_func}{A vector of objective functions collected through the algorithm iterations}
%% ...
}
%%\references{
%% ~put references to the literature/web site here ~
%%}
\author{
Leyuan Qian
}
%%\note{
%%  ~~further notes~~
%%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%%}
\examples{
# generate a random array with missing data
dims <- c(24, 3, 207)
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

# run SmoothHOOI
res <- mglram(tnsr = tnsr, ranks = c(3, 2), init=0, D = D2,
       lambda = 5, max_iter = 500, tol = 1e-5, L0 = NULL)

res$conv
res$L
res$R
res$G[ , , 1]
res$est[ , , 1]
res$obj_func[1:10]

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
