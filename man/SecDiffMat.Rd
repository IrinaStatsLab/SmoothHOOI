\name{SecDiffMat}
\alias{SecDiffMat}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Generation of a second-order difference matrix with circular nature
}
\description{
Generate of a second-order difference matrix with circular nature
}
\usage{
SecDiffMat(dim)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{dim}{
  The dimension of the matrix, a.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{D2}{A second-order matrix of dimension a*a}
%%  \item{comp2 }{Description of 'comp2'}
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
  SecDiffMat(24)
  
  SecDiffMat(10)
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
