\name{kcv}
\alias{kcv}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Selection of optimal hyperparameters with K-fold cross-validation
}
\description{
Find the optimal hyperparameters that minimize the cross-validation (CV) error in a certain search grid.
}
\usage{
kcv(tnsr, rank_grid, lambda_seq, k, L0_, D, tol = 0.1, max_iter = 500L, init = 0)
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
  \item{L0_}{
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
