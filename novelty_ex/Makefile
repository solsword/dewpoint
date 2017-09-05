.PHONY: clean

clean:
	rm data/common/*.png
	rm data/rare/*.png

.PHONY: data

data: data/common/h-p-N.png \
	    data/common/g-p-N.png \
			data/common/h-g-N.png \
			data/common/p-g-N.png \
			data/rare/p-p-N.png \
			data/rare/g-g-N.png \
			data/rare/h-p-F.png \
			data/rare/b-p-N.png \
			data/rare/b-p-F.png \
			data/rare/g-g-F.png

data/%-N.png:
	$(eval P1 := $(shell basename $* | cut -d- -f1))
	$(eval P2 := $(shell basename $* | cut -d- -f2))
	convert data/${P1}h-small.png data/${P2}b-small.png -append $@

data/%-F.png:
	$(eval P1 := $(shell basename $* | cut -d- -f1))
	$(eval P2 := $(shell basename $* | cut -d- -f2))
	convert data/${P1}h-small.png -flop data/${P2}b-small.png -append $@

.PHONY: run

run: ex.py data
	optirun ./ex.py
