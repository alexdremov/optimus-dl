variable "VERSION" {
  default = "latest"
}

variable "ARCH" {
  default = "amd64"
}

variable "CCACHE_SRC" {
  default = "target:ccache-empty"
}

group "default" {
  targets = [ "optimus-dl", "optimus-dl-interactive" ]
}

target "ccache-empty" {
  dockerfile-inline = "FROM scratch"
}

target "optimus-dl" {
  context = "."
  target = "base"
  dockerfile = "./docker/Dockerfile"
  tags = [
    "alexdremov/optimus-dl:${VERSION}",
    "alexdremov/optimus-dl:latest",
  ]
  platform = "linux/${ARCH}"
  args = {
    VERSION = "${VERSION}"
    ARCH = "${ARCH}"
  }
  contexts = {
    ccache_src = "${CCACHE_SRC}"
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
  platform = "linux/${ARCH}"
  args = {
    VERSION = "${VERSION}"
    ARCH = "${ARCH}"
  }
  contexts = {
    ccache_src = "${CCACHE_SRC}"
  }
}

target "ccache-export" {
  context = "."
  target = "ccache-export"
  dockerfile = "./docker/Dockerfile"
  platform = "linux/${ARCH}"
  args = {
    ARCH = "${ARCH}"
  }
  contexts = {
    ccache_src = "${CCACHE_SRC}"
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
