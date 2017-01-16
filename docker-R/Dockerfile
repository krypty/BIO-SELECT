FROM r-base@sha256:287d8b76535622e3693ea799f2e8fc31f7845111954b14ac1ef406deb4e811b9

# needed for biobase
RUN apt-get update && apt-get install -y libssl-dev libxml2-dev libcurl4-gnutls-dev

RUN echo 'source("https://bioconductor.org/biocLite.R")' > /tmp/packages.R && \
    echo 'biocLite("Biobase")' >> /tmp/packages.R && \
    echo 'biocLite("GEOquery")' >> /tmp/packages.R && \
    echo 'biocLite("limma")' >> /tmp/packages.R && \
    echo 'biocLite("golubEsets")' >> /tmp/packages.R && \

    Rscript /tmp/packages.R

ADD limma-*.R ./

VOLUME dataset

#CMD ["Rscript", "--no-save", "--no-restore", "--verbose",  "limma-mile.R"]
#CMD ["Rscript", "--no-save", "--no-restore", "--verbose",  "limma-golub.R"]
CMD ["/bin/bash"]
