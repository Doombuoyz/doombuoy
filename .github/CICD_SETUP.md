# GitHub Actions CI/CD Setup Guide

This repository uses GitHub Actions to automatically build and publish the package to Test PyPI whenever code is pushed to the `master` or `main` branch.

## 🔧 Setup Instructions

### Step 1: Add Test PyPI API Token to GitHub Secrets

1. Go to your GitHub repository: `https://github.com/Doombuoyz/doombuoy`
2. Click on **Settings** (repository settings, not your account)
3. In the left sidebar, click on **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add the following secret:
   - **Name**: `TEST_PYPI_API_TOKEN`
   - **Value**: `your api key`
6. Click **Add secret**

### Step 2: Enable Workflow Permissions (CRITICAL)

**This step is REQUIRED** - without it, you'll get "Resource not accessible by integration" error.

1. Go to: `https://github.com/Doombuoyz/doombuoy/settings/actions`
2. Scroll down to **"Workflow permissions"**
3. Select **"Read and write permissions"** (NOT "Read repository contents and packages permissions")
4. Check ✅ **"Allow GitHub Actions to create and approve pull requests"**
5. Click **"Save"**

### Step 3: Push Your Code

Once the secret is added and permissions are configured, the workflow will automatically trigger on every push to `master` or `main` branch.

## 📋 What the Pipeline Does

### On Push to `master`/`main`:

1. **Checks out** your code
2. **Sets up** Python 3.11 environment
3. **Installs** Poetry
4. **Increments** the patch version (e.g., 2025.0.0.25 → 2025.0.0.26)
5. **Commits** the version bump back to the repository with `[skip ci]` to avoid infinite loops
6. **Installs** all dependencies
7. **Builds** the package using `poetry build`
8. **Publishes** to Test PyPI using `poetry publish -r test-pypi`
9. **Creates** a GitHub release with the new version tag

### On Pull Requests or Feature Branches:

- Runs tests across Python 3.11 and 3.12
- Checks code style with flake8
- Does NOT publish (test-only workflow)

## 🎯 Version Increment Strategy

The pipeline automatically increments the **patch** version:
- `2025.0.0.25` → `2025.0.0.26` → `2025.0.0.27` ...

To manually change version type:
```bash
# Minor version bump: 2025.0.0 → 2025.1.0
poetry version minor

# Major version bump: 2025.0.0 → 2026.0.0
poetry version major

# Custom version
poetry version 2025.1.0
```

Then commit and push - the pipeline will continue from that version.

## 🚀 Usage

### Automatic Publishing (Recommended)

Simply commit and push your changes:

```bash
git add .
git commit -m "Add new feature"
git push origin master
```

The pipeline will automatically:
- Bump version
- Build package
- Publish to Test PyPI
- Create GitHub release

### Manual Publishing (If Needed)

If you need to publish manually:

```bash
# Increment version
poetry version patch

# Build
poetry build

# Configure Test PyPI
poetry config repositories.test-pypi https://test.pypi.org/legacy/

# Publish
poetry publish -r test-pypi
```

## 📊 Monitoring

- View workflow runs: `https://github.com/Doombuoyz/doombuoy/actions`
- Check published versions: `https://test.pypi.org/project/doombuoy/`
- View releases: `https://github.com/Doombuoyz/doombuoy/releases`

## ⚠️ Important Notes

1. **[skip ci]** in commit messages prevents infinite loops
2. The pipeline requires **write permissions** for:
   - Committing version bumps
   - Creating releases
3. Ensure your repository has **Actions** enabled (Settings → Actions → General)
4. The workflow runs on **Linux** (ubuntu-latest) for consistency

## 🔒 Security

- API tokens are stored as **encrypted secrets** in GitHub
- Never commit API tokens directly to your repository
- Tokens are only exposed to the workflow during execution

## 🛠️ Customization

### Change Version Bump Type

Edit `.github/workflows/publish.yml`, line with `poetry version patch`:

```yaml
# For minor version bumps
poetry version minor

# For major version bumps  
poetry version major
```

### Add Pre-Release Versions

```yaml
poetry version prerelease
```

### Trigger on Tags Instead of Push

Replace the `on:` section:

```yaml
on:
  push:
    tags:
      - 'v*.*.*'
```

## 📝 Troubleshooting

### "Resource not accessible by integration" Error

This error means GitHub Actions doesn't have permission to create releases or push commits:

**Solution:**
1. Go to `Settings → Actions → General`
2. Under "Workflow permissions", select **"Read and write permissions"**
3. Check **"Allow GitHub Actions to create and approve pull requests"**
4. Save and re-run the workflow

### Pipeline Fails on First Run

- Ensure `TEST_PYPI_API_TOKEN` secret is added correctly
- Check that your Test PyPI token hasn't expired
- Verify the token has upload permissions
- Confirm workflow permissions are set to "Read and write"

### Version Conflicts

If you see "File already exists" error:
- Test PyPI doesn't allow re-uploading the same version
- The pipeline will auto-increment, so this shouldn't happen
- If it does, manually bump the version and push again

### Permission Errors

Ensure GitHub Actions has write permissions:
1. Go to Settings → Actions → General
2. Under "Workflow permissions", select "Read and write permissions"
3. Save changes

## 🎉 Success

Once configured, every push to master will:
- ✅ Auto-increment version
- ✅ Build package
- ✅ Publish to Test PyPI
- ✅ Create GitHub release
- ✅ Update repository with new version

No manual intervention needed!
