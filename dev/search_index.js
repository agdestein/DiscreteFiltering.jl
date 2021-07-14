var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = DiscreteFiltering","category":"page"},{"location":"#DiscreteFiltering","page":"Home","title":"DiscreteFiltering","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DiscreteFiltering.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [DiscreteFiltering]","category":"page"},{"location":"#DiscreteFiltering.DiscreteFiltering","page":"Home","title":"DiscreteFiltering.DiscreteFiltering","text":"Discrete filtering toolbox\n\n\n\n\n\n","category":"module"},{"location":"#DiscreteFiltering.AbstractIntervalDomain","page":"Home","title":"DiscreteFiltering.AbstractIntervalDomain","text":"ClosedIntervalDomain\n\nAbstract type for interval domains.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.AdvectionEquation","page":"Home","title":"DiscreteFiltering.AdvectionEquation","text":"AdvectionEquation(domain, filter = IdentityFilter())\n\nFiltered advection equation.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.BurgersEquation","page":"Home","title":"DiscreteFiltering.BurgersEquation","text":"BurgersEquation(domain, filter = IdentityFilter())\n\nFiltered Burgers equation.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.ClosedIntervalDomain","page":"Home","title":"DiscreteFiltering.ClosedIntervalDomain","text":"ClosedIntervalDomain(left, right)\n\nInterval domain.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.ConvolutionalFilter","page":"Home","title":"DiscreteFiltering.ConvolutionalFilter","text":"ConvolutionalFilter(kernel)\n\nConvolutional filter, parameterized by a filter kernel.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.DiffusionEquation","page":"Home","title":"DiscreteFiltering.DiffusionEquation","text":"DiffusionEquation(\n    domain,\n    filter = IdentityFilter(),\n    f = nothing,\n    g_a = nothing,\n    g_b = nothing\n)\n\nFiltered diffusion equation.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.Domain","page":"Home","title":"DiscreteFiltering.Domain","text":"Domain\n\nAbstract type for different domains.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.Equation","page":"Home","title":"DiscreteFiltering.Equation","text":"Abstract equation.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.Filter","page":"Home","title":"DiscreteFiltering.Filter","text":"Abstract continuous filter.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.IdentityFilter","page":"Home","title":"DiscreteFiltering.IdentityFilter","text":"IdentityFilter()\n\nIdentity filter, which does not filter.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.PeriodicIntervalDomain","page":"Home","title":"DiscreteFiltering.PeriodicIntervalDomain","text":"PeriodicIntervalDomain(left, right)\n\nPeriodic interval domain.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.TopHatFilter","page":"Home","title":"DiscreteFiltering.TopHatFilter","text":"TopHatFilter(width)\n\nTop hat filter, parameterized by a variable filter width.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.advection_matrix-Tuple{DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.advection_matrix","text":"advection_matrix(domain, n)\n\nAssemble discrete advection matrix.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.diffusion_matrix-Tuple{DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.diffusion_matrix","text":"diffusion_matrix(domain, n)\n\nAssemble discrete diffusion matrix.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.discretize-Tuple{DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.discretize","text":"discretize(domain, n)\n\nDiscretize domain with n points.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.filter_matrix-Tuple{DiscreteFiltering.Filter, DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.filter_matrix","text":"filter_matrix(f, domain, n)\n\nAssemble discrete filtering matrix from a continuous filter f.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.filter_matrix_meshwidth-Tuple{TopHatFilter, PeriodicIntervalDomain, Any}","page":"Home","title":"DiscreteFiltering.filter_matrix_meshwidth","text":"filter_matrix_meshwidth(f, domain, n)\n\nAssemble discrete filtering matrix from a continuous filter f width constant width h(x) = Delta x  2.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.gaussian-Tuple{Any}","page":"Home","title":"DiscreteFiltering.gaussian","text":"gaussian(σ²) -> Function\n\nCreate Gaussian function with variance σ².\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.inverse_filter_matrix-Tuple{DiscreteFiltering.Filter, DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.inverse_filter_matrix","text":"inverse_filter_matrix(f, domain, n)\n\nApproximate inverse of discrete filtering matrix, given filter f.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.inverse_filter_matrix_meshwidth-Tuple{TopHatFilter, PeriodicIntervalDomain, Any}","page":"Home","title":"DiscreteFiltering.inverse_filter_matrix_meshwidth","text":"inverse_filter_matrix_meshwidth(f, domain, n)\n\nAssemble inverse discrete filtering matrix from a continuous filter f width constant width h(x) = Delta x  2.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.solve-Tuple{DiscreteFiltering.Equation, Any, Any, Any}","page":"Home","title":"DiscreteFiltering.solve","text":"solve(equation, u, tlist, n; method = \"filterfirst\")\n\nSolve equation from tlist[1] to tlist[2] with initial conditions u and a discretization of n points. If method is \"filterfirst\", the equation is filtered then discretized. If method is \"discretizefirst\", the equation is discretized then filtered.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.solve_burgers-Tuple{}","page":"Home","title":"DiscreteFiltering.solve_burgers","text":"solve_burgers()\n\nSolve Burgers equation.\n\n\n\n\n\n","category":"method"}]
}
