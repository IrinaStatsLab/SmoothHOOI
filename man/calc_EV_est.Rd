\name{calc_EV_est}
\alias{calc_EV_est}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Calculate the proportion of total explained variability explained by the fit
}
\description{
Calculate the proportion of total explained variability explained by the fit obtained from SmoothHOOI result (est). 
}
\usage{
calc_EV_est(M_denom, est)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{M_denom}{
A multidimensional array representing the original tensor fitted by SmoothHOOI, can include missing data. 
}
  \item{est}{
Fitted data given by SmoothHOOI. 
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
The proportion of total variability explained by the fit.
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
%
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
