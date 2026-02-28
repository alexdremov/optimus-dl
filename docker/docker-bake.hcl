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
  platforms = ["linux/amd64"]
  args = {
    VERSION = "${VERSION}"
    ARCH = "${ARCH}"
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
  platforms = ["linux/amd64"]
  args = {
    VERSION = "${VERSION}"
    ARCH = "${ARCH}"
  }
}

target "ccache-export" {
  context = "."
  target = "ccache-only"
  dockerfile = "./docker/Dockerfile"
  tags = ["alexdremov/optimus-dl-ccache:${ARCH}"]
  args = {
    ARCH = "${ARCH}"
  }
}

target "ccache-seed" {
  context = "."
  target = "ccache-seed"
  dockerfile = "./docker/Dockerfile"
  args = {
    ARCH = "${ARCH}"
  }
}
