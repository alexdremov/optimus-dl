variable "VERSION" {
  default = "latest"
}

group "default" {
  targets = [ "optimus-dl", "optimus-dl-interactive" ]
}

target "optimus-dl" {
  context = "."
  target = "base"
  dockerfile = "./docker/Dockerfile"
  tags = [
    "alexdremov/optimus-dl:${VERSION}",
    "alexdremov/optimus-dl:latest",
  ]
  platforms = [
    "linux/amd64",
    "linux/arm64"
  ]
  args = {
    VERSION = "${VERSION}"
  }
}

target "optimus-dl-interactive" {
  context = "."
  target = "interactive"
  dockerfile = "./docker/Dockerfile"
  tags = [
    "alexdremov/optimus-dl:interactive-${VERSION}",
    "alexdremov/optimus-dl:interactive-latest",
  ]
  platforms = [
    "linux/amd64",
    "linux/arm64"
  ]
  args = {
    VERSION = "${VERSION}"
  }
}
