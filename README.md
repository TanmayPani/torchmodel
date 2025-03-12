# Torchmodel

## Recommended usage:

Easiest usage is with the [uv package manager](https://docs.astral.sh/uv/#uv). 
1. Install uv in your system using instructionns [here](https://docs.astral.sh/uv/getting-started/installation/).
2. Create a uv project using `uv init <project name>`
3. Inside the directory that uv just created, create another directory `packages`
4. Edit the `pyproject.toml` file to include all folders in `packages` directory as workspace members, by adding:
   ```
   [tool.uv.workspace]
    members = ["packages/*"]
   ```
5. Inside the `packages` directory, clone this repo
   ```
   cd packages
   git clone https://github.com/TanmayPani/torchmodel
   cd ..
   ```
6. Add `torchmodel` to the dependency list in your `pyproject.toml`
   ```
   [project]
   dependencies = [torchmodel, <any other dependencies your project might have...>]
   ```
7. Set the source of this dependency in `[tool.uv.sources]` field of `pyproject.toml`
   ```
   [tool.uv.sources]
   torchmodel = {workspace = true}
   ```
8. Use this in your projects `main.py` or any other `.py` file in the project root by
   ```
   import torchmodel
   ```
   or
   ```
   from torchmodel import torchmodel, datasets, callbacks, archs #skip whatever you
                                                                 #dont need,
                                                                 #add whatever you do
   ```

