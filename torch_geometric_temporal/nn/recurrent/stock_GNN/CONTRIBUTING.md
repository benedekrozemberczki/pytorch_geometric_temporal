# How to contribute

I'm really glad you're reading this, because we need volunteer developers to help this project come to fruition.

Here are some important things to check when you contribute:

  * Please make sure that you write tests.
  * Update the documentation.
  * Add the new model to the readme.
  * If your contribution is a paper please update the resource documentation file. 
  
## Testing


PyTorch Geometric Temporal's testing is located under `test/`.
Run the entire test suite with

```
python setup.py test
```

## Submitting changes

Please send a [GitHub Pull Request to PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/pull/new/master) with a clear list of what you've done (read more about [pull requests](http://help.github.com/pull-requests/)). Please follow our coding conventions (below) and make sure all of your commits are atomic (one feature per commit).

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
Benedek

