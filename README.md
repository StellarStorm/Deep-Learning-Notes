# Deep Learning Notes

This is the source for my online Deep Learning notebook

## Using

1. [Install Hugo](https://gohugo.io/overview/installing/)
2. Clone this repository

```bash
git clone https://github.com/StellarStorm/Deep-Learning-Notes.git
cd Deep-Learning-Notes
```

3. Install Anatole theme

```bash
git submodule add -f https://github.com/lxndrblz/anatole.git  themes/anatole
```

4. Run Hugo.

```bash
hugo server
```

The Anatole theme will be started by default. To use a different theme, install
as above and then run `hugo server -t NEWTHEME`

5. Under `/content/` this repository contains the following:

- A section called `/post/` with all published notes. Upcoming posts should
  be written here in Markdown.
- A headless bundle called `homepage` that you may want to use for single page
  applications. You can find instructions about headless bundles over
  [here](https://gohugo.io/content-management/page-bundles/#headless-bundle)
- An `about.md` that provides the `/about/` page


## Deploying to GitHub

1. Set up GitHub repository
2. Edit `baseURL` in config.toml (if needed) to point to the future GitHub Pages
   URL for this site
3. Commit and push the local repository to GitHub
4. Wait for the GitHub Actions deployment to complete (this builds the site on
   a separate gh-pages branch)
5. Under repository settings > Pages, change the branch to "gh-pages" and the
   folder to "/(root)"
