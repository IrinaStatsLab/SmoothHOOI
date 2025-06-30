\name{MakeIdent}
\alias{MakeIdent}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Identifiability correction
}
\description{
Matrix rotation to ensure identifiability of L, R, and G in the downstream analysis
}
\usage{
MakeIdent(L, G, R)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{L}{
  A matrix L of dimension a*r1 generated from mglram(). 
}
  \item{G}{
  An array G of dimension r1*r2*n generated from mglram(). 
}
  \item{R}{
  A matrix R of dimension b*r2 generated from mglram(). 
}
}
\details{
This function is developed based on the rTensor package (Li et al., 2018). 
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{L_tilde}{L matrix after identifiability correction}
\item{R_tilde}{R matrix after identifiability correction}
\item{G_tilde}{G array after identifiability correction}
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
