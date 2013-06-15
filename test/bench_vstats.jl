# Benchmarks on vstats

using MLBase

macro bench_vstats(name, f0, f1, rp, a)
	quote
		println("Bench: $($name)")

		# warming up
		($f0)($a)
		($f1)($a)

		# main
		t0 = @elapsed for i in 1 : $rp
			($f0)($a)
		end

		t1 = @elapsed for i in 1 : $rp
			($f1)($a)
		end

		@printf("\tJulia function:  %6.4f sec\n", t0)
		@printf("\tMLBase vstats:   %6.4f sec | gain = %6.4fx\n", t1, t0/t1)

		println()
	end
end


x = rand(2000, 2000)

max1_jfun(x::Matrix) = max(x, (), 1)
max1_vfun(x::Matrix) = vmax(x, 1)
@bench_vstats("colwise-max", max1_jfun, max1_vfun, 10, x)

max2_jfun(x::Matrix) = max(x, (), 2)
max2_vfun(x::Matrix) = vmax(x, 2)
@bench_vstats("rowwise-max", max2_jfun, max2_vfun, 10, x)

min1_jfun(x::Matrix) = min(x, (), 1)
min1_vfun(x::Matrix) = vmin(x, 1)
@bench_vstats("colwise-min", min1_jfun, min1_vfun, 10, x)

min2_jfun(x::Matrix) = min(x, (), 2)
min2_vfun(x::Matrix) = vmin(x, 2)
@bench_vstats("rowwise-min", min2_jfun, min2_vfun, 10, x)

sum1_jfun(x::Matrix) = sum(x, 1)
sum1_vfun(x::Matrix) = vsum(x, 1)
@bench_vstats("colwise-sum", sum1_jfun, sum1_vfun, 10, x)

sum2_jfun(x::Matrix) = sum(x, 2)
sum2_vfun(x::Matrix) = vsum(x, 2)
@bench_vstats("rowwise-sum", sum2_jfun, sum2_vfun, 10, x)

amax1_jfun(x::Matrix) = max(abs(x), (), 1)
amax1_vfun(x::Matrix) = vamax(x, 1)
@bench_vstats("colwise-amax", amax1_jfun, amax1_vfun, 10, x)

amax2_jfun(x::Matrix) = max(abs(x), (), 2)
amax2_vfun(x::Matrix) = vamax(x, 2)
@bench_vstats("rowwise-amax", amax2_jfun, amax2_vfun, 10, x)

amin1_jfun(x::Matrix) = min(abs(x), (), 1)
amin1_vfun(x::Matrix) = vamin(x, 1)
@bench_vstats("colwise-amin", amin1_jfun, amin1_vfun, 10, x)

amin2_jfun(x::Matrix) = min(abs(x), (), 2)
amin2_vfun(x::Matrix) = vamin(x, 2)
@bench_vstats("rowwise-amin", amin2_jfun, amin2_vfun, 10, x)

asum1_jfun(x::Matrix) = sum(abs(x), 1)
asum1_vfun(x::Matrix) = vasum(x, 1)
@bench_vstats("colwise-asum", asum1_jfun, asum1_vfun, 10, x)

asum2_jfun(x::Matrix) = sum(abs(x), 2)
asum2_vfun(x::Matrix) = vasum(x, 2)
@bench_vstats("rowwise-asum", asum2_jfun, asum2_vfun, 10, x)

sqsum1_jfun(x::Matrix) = sum(abs2(x), 1)
sqsum1_vfun(x::Matrix) = vsqsum(x, 1)
@bench_vstats("colwise-sqsum", sqsum1_jfun, sqsum1_vfun, 10, x)

sqsum2_jfun(x::Matrix) = sum(abs2(x), 2)
sqsum2_vfun(x::Matrix) = vsqsum(x, 2)
@bench_vstats("rowwise-sqsum", sqsum2_jfun, sqsum2_vfun, 10, x)




