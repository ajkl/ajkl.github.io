---
layout:     post
title:      "Download files multi-threaded in R"
subtitle:   ""
date:       2015-07-22 12:00:00
author:     ""
header-img: "img/post-bg-05.jpg"
---

I had a csv with 5 columns with one of the columns having image urls (picture_url) and one of them having the image id (id).
And I had 100,000 of these.
Task was to download all the images with image-id as the image name.

I tried regular download.file() in R but that soon turned out to be an epic fail.
Hello doParallel!

{% highlight r %}
image.df <- read.table("images.csv", 
                      header = TRUE, 
                      sep = ",", 
                      colClasses = c(rep("character", 5)))
{% endhighlight %}

You can find the number of cores on linix using [lscpu](http://manpages.ubuntu.com/manpages/saucy/man1/lscpu.1.html) ```lscpu```
Register the cores with doParallel

{% highlight r %}
require(doParallel)
registerDoParallel(cores=24)
{% endhighlight %}

Business end of things

{% highlight r %}
foreach(i=1:nrow(image.df)) %dopar% try(download.file(image.df[i,'picture_url'], 
                                                      destfile=paste(image.df[i,'id'], 
                                                                    ".jpg", 
                                                                    sep="")
                                                      )
                                        )
{% endhighlight %}

I am pretty sure there are other (and probably even better) ways to achieve this, but this gets the job done pretty nicely for me.
