# build the docker image
.PHONY: image
image:
	docker build --progress=plain -t abstraqt .

.PHONY: run
run: image
	mkdir -p home
	docker run \
		-it \
		--rm \
		-v $$(pwd)/..:/abstraqt \
		-v $$(pwd)/home/.cache:/root/.cache \
		abstraqt \
			/bin/bash

.PHONY: clean
clean:
	rm -r home
