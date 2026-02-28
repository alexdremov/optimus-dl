variable "VERSION" {
  default = "latest"
}

variable "ARCH" {
  default = "amd64"
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
  args = {
    VERSION = "${VERSION}"
    ARCH = "${ARCH}"
  }
  contexts = {
    ccache_src = "target:ccache-only"
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
  args = {
    VERSION = "${VERSION}"
    ARCH = "${ARCH}"
  }
  contexts = {
    ccache_src = "target:ccache-only"
  }
}

target "ccache-export" {
  context = "."
  target = "ccache-export"
  dockerfile = "./docker/Dockerfile"
  args = {
    ARCH = "${ARCH}"
  }
  contexts = {
    ccache_src = "target:ccache-only"
  }
}

target "ccache-only" {
  context = "."
  target = "ccache-only"
  dockerfile = "./docker/Dockerfile"
  args = {
    ARCH = "${ARCH}"
  }
}
