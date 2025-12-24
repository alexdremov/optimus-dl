variable "VERSION" {
  default = "latest"
}

group "default" {
  targets = [ "optimus-dl" ]
}

target "optimus-dl" {
  context = "."
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
