.PHONY: clean

clean:
	rm data/*/*.png

.PHONY: data

data: data/train/tomato-khaki-black.png \
	    data/train/skyblue-violet-black.png \
			data/train/pink-purple-white.png \
			data/train/darkblue-teal-white.png \
			data/test/skyblue-violet-white.png \
			data/test/pink-purple-black.png \
			data/test/khaki-tomato-black.png \
			data/test/teal-darkblue-white.png \
			data/test/khaki-tomato-white.png \
			data/test/teal-darkblue-black.png

data/train/%.png:
	$(eval C1 := $(shell echo $* | cut -d- -f1))
	$(eval C2 := $(shell echo $* | cut -d- -f2))
	$(eval C3 := $(shell echo $* | cut -d- -f3))
	convert \
	  \( 'xc:${C1}[16x16!]' 'xc:${C2}[16x16!]' +append \) \
		'xc:${C3}[32x16!]' -append \
		$@

data/test/%.png:
	$(eval C1 := $(shell echo $* | cut -d- -f1))
	$(eval C2 := $(shell echo $* | cut -d- -f2))
	$(eval C3 := $(shell echo $* | cut -d- -f3))
	convert \
	  \( 'xc:${C1}[16x16!]' 'xc:${C2}[16x16!]' +append \) \
		'xc:${C3}[32x16!]' -append \
		$@


.PHONY: run

run: ex.py data
	optirun ./ex.py
