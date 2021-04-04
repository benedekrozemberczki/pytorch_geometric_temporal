# How to contribute

I'm really glad you're reading this, because we need volunteer developers to help this project come to fruition.

If you haven't already, come find us in IRC ([#opengovernment](irc://chat.freenode.net/opengovernment) on freenode). We want you working on things you're excited about.

Here are some important resources:

  * [OpenGovernment for Developers](http://opengovernment.org/pages/developer) tells you where we are,
  * [Our roadmap](http://opengovernment.org/pages/wish-list) is the 10k foot view of where we're going, and
  * [Pivotal Tracker](http://pivotaltracker.com/projects/64842) is our day-to-day project management space.
  * Mailing list: Join our [developer list](http://groups.google.com/group/opengovernment/)
  * Bugs? [Lighthouse](https://participatorypolitics.lighthouseapp.com/projects/47665-opengovernment/overview) is where to report them
  * IRC: chat.freenode.net channel [#opengovernment](irc://chat.freenode.net/opengovernment). We're usually there during business hours.

## Testing

We have a handful of Cucumber features, but most of our testbed consists of RSpec examples. Please write RSpec examples for new code you create.

## Submitting changes

Please send a [GitHub Pull Request to PyTorch Geometric Temporal](https://github.com/opengovernment/opengovernment/pull/new/master) with a clear list of what you've done (read more about [pull requests](http://help.github.com/pull-requests/)). Please follow our coding conventions (below) and make sure all of your commits are atomic (one feature per commit).

Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

    $ git commit -m "A brief summary of the commit
    > 
    > A paragraph describing what changed and its impact."

## Coding conventions

Start reading our code and you'll get the hang of it. We optimize for readability:

  * We write tests for the data loaders, iterators and layers.
  * We use the type hinting feature of Python.
  * We avoid the uses of public methods and vaiarbles in the classes.
  * Hyperparameters belong to the constructors.
  * Auxiliiary layer instances should have long names.
  * Make linear algebra operations line-by-line.

Thanks,
Benedek Rozemberczki

