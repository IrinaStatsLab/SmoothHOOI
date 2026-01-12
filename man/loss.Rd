\name{loss}
\alias{loss}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Calculation of losses of M and L
}
\description{
Compare the original data and its reconstruction with calculation of losses of M and L
}
\usage{
loss(tnsr, smooth_tnsr, L, true_L, R, true_R)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tnsr}{
  A multi-dimensional array of dimension a*b*n. Should be organized so that the first two dimensions are to be compressed.
}
  \item{smooth_tnsr}{
  A multi-dimensional array of dimension a*b*n, which contains the smooth, complete ground-truth data.
}
  \item{L}{
  Estimated L matrix. 
}
  \item{true_L}{
  Ground-truth L matrix.
}
 \item{R}{
  Estimated R matrix. 
}
  \item{true_R}{
  Ground-truth R matrix.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{loss_M}{Losses in the reconstruction of M}
\item{loss_L}{Losses in the reconstruction of L}
\item{loss_R}{Losses in the reconstruction of R}
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
