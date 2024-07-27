# Use the official R base image
FROM rocker/r-ver:4.2.2

# Install the Linux libraries needed for plumber
RUN apt-get update -qq && apt-get install -y libssl-dev libcurl4-gnutls-dev libpng-dev

# Install R packages not included in r-base
RUN R -e "install.packages(c('plumber', 'tidyverse', 'caret', 'ranger', 'Metrics'))"

# Files to copy and to where
COPY API.R /API.R
COPY diabetes_binary_health_indicators_BRFSS2015.csv /diabetes_binary_health_indicators_BRFSS2015.csv

# Expose port 8000
EXPOSE 8000

# Run the API locally at port 8000
ENTRYPOINT ["R", "-e", "pr <- plumber::plumb('/API.R'); pr$run(host = '0.0.0.0', port = 8000)"]
