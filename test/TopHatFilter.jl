@testset "Top hat filter" begin
    h(x) = 1 - 1 / 2 * cos(x)
    f = TopHatFilter(h)
    @test f.width == h
end
