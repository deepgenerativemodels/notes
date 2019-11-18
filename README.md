# CS236 Notes: Deep Generative Models

These notes form a concise introductory course on deep generative models. They are based on Stanford [CS236](https://deepgenerativemodels.github.io/), taught by [Aditya Grover](http://aditya-grover.github.io/) and [Stefano Ermon](http://cs.stanford.edu/~ermon/), and have been written by [Aditya Grover](http://aditya-grover.github.io/), with the [help](https://github.com/deepgenerativemodels/notes/commits/master) of many students and course staff.

The compiled notes are available [here](https://deepgenerativemodels.github.io/notes/index.html).

# Contributing

This material is under construction! Please help us resolve typos by submitting PRs to this repo.

## Compilation

The notes are written in Markdown and are compiled into HTML using Jekyll. Please add your changes directly to the Markdown source code. In order to install jekyll, you can follow the instructions posted on their website (https://jekyllrb.com/docs/installation/).

To compile Markdown to HTML, run the following commands from the root of your repo:

1) rm -r docs/
2) jekyll serve  # This should create a folder called _site. Note: This creates a running server; press Ctrl-C to stop the server before proceeding
3) git add {...} # Add changed files here
4) git commit -am "your commit message describing what you did"
5) git push origin master

## Notes on building the site on Windows

Note that jekyll is only supported on GNU/Linux, Unix, or macOS. Thus, if you run Windows 10 on your local machine, you will have to install Bash on Ubuntu on Windows. Windows gives instructions on how to do that <a href="https://docs.microsoft.com/en-us/windows/wsl/install-win10">here</a> and Jekyll's <a href="https://jekyllrb.com/docs/windows/">website</a> offers helpful instructions on how to proceed through the rest of the process.

## Notes on Github permissions

Note that if you cloned the ermongroup/cs228-notes repo directly you may see an error like "remote: Permission to ermongroup/cs228-notes.git denied to userjanedoe". If that is the case, then you need to fork this repo first. Then, if your github profile were userjanedoe, you would need to first push your local updates to your forked repo like so:

git push https://github.com/deepgenerativemodels/notes.git master

And then you could go and submit the pull request through the GitHub website.
