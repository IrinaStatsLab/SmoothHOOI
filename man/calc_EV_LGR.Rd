\name{calc_EV_LGR}
\alias{calc_EV_LGR}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Calculate incremental and accumulative explained variability
}
\description{
Calculate incremental and accumulative explained variability from SmoothHOOI result (identifiability corrected L, G, R from 2-way compression). 
}
\usage{
calc_EV_LGR(M_denom, L_tilde, G_tilde, R_tilde)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{M_denom}{
A multidimensional array representing the original tensor fitted by SmoothHOOI, can include missing data. 
}
  \item{L_tilde}{
L after identifiability correction.
}
  \item{G_tilde}{
G after identifiability correction.
}
  \item{R_tilde}{
R after identifiability correction.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
\item{cum_EV_L}{Acummulative explained variability when increasing the number of L components but keeping R fixed}
\item{sep_EV_L}{Incremental explained variability given by each L component when R is fixed}
\item{cum_EV_R}{Acummulative explained variability when increasing the number of R components but keeping L fixed}
\item{sep_EV_R}{Incremental explained variability given by each R component when L is fixed}
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
}
%%\examples{
%%}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
