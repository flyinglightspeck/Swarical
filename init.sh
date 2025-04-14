docker buildx build --platform linux/amd64 -t swarical .

docker run -p 8888:8888 swarical
