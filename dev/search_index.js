var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = DiscreteFiltering","category":"page"},{"location":"#DiscreteFiltering","page":"Home","title":"DiscreteFiltering","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DiscreteFiltering.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [DiscreteFiltering]","category":"page"},{"location":"#DiscreteFiltering.DiscreteFiltering","page":"Home","title":"DiscreteFiltering.DiscreteFiltering","text":"Discrete filtering toolbox\n\n\n\n\n\n","category":"module"},{"location":"#DiscreteFiltering.AbstractIntervalDomain","page":"Home","title":"DiscreteFiltering.AbstractIntervalDomain","text":"ClosedIntervalDomain\n\nAbstract type for interval domains.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.ClosedIntervalDomain","page":"Home","title":"DiscreteFiltering.ClosedIntervalDomain","text":"ClosedIntervalDomain(left, right)\n\nInterval domain.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.ConvolutionalFilter","page":"Home","title":"DiscreteFiltering.ConvolutionalFilter","text":"ConvolutionalFilter(kernel)\n\nConvolutional filter, parameterized by a filter kernel.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.Domain","page":"Home","title":"DiscreteFiltering.Domain","text":"Domain\n\nAbstract type for different domains.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.Filter","page":"Home","title":"DiscreteFiltering.Filter","text":"Abstract continuous filter.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.PeriodicIntervalDomain","page":"Home","title":"DiscreteFiltering.PeriodicIntervalDomain","text":"PeriodicIntervalDomain(left, right)\n\nPeriodic interval domain.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.TopHatFilter","page":"Home","title":"DiscreteFiltering.TopHatFilter","text":"TopHatFilter(width)\n\nTop hat filter, parameterized by a variable filter width.\n\n\n\n\n\n","category":"type"},{"location":"#DiscreteFiltering.advection_matrix-Tuple{DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.advection_matrix","text":"advection_matrix(domain, n)\n\nAssemble discrete advection matrix.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.diffusion_matrix-Tuple{DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.diffusion_matrix","text":"diffusion_matrix(n)\n\nAssemble discrete diffusion matrix.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.discretize-Tuple{DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.discretize","text":"discretize(domain, n)\n\nDiscretize domain with n points.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.filter_matrix-Tuple{DiscreteFiltering.Filter, DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.filter_matrix","text":"filter_matrix(f, domain, n)\n\nAssemble discrete filtering matrix from a continuous filter f.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.filter_matrix_meshwidth-Tuple{TopHatFilter, PeriodicIntervalDomain, Any}","page":"Home","title":"DiscreteFiltering.filter_matrix_meshwidth","text":"filter_matrix_meshwidth(f, domain, n)\n\nAssemble discrete filtering matrix from a continuous filter f width constant width h(x) = Delta x  2.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.inverse_filter_matrix-Tuple{DiscreteFiltering.Filter, DiscreteFiltering.Domain, Any}","page":"Home","title":"DiscreteFiltering.inverse_filter_matrix","text":"inverse_filter_matrix(f, domain, n)\n\nApproximate inverse of discrete filtering matrix, given filter f.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.inverse_filter_matrix_meshwidth-Tuple{TopHatFilter, PeriodicIntervalDomain, Any}","page":"Home","title":"DiscreteFiltering.inverse_filter_matrix_meshwidth","text":"inverse_filter_matrix_meshwidth(f, domain, n)\n\nAssemble inverse discrete filtering matrix from a continuous filter f width constant width h(x) = Delta x  2.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.solve_advection-Tuple{}","page":"Home","title":"DiscreteFiltering.solve_advection","text":"solve_advection()\n\nSolve advection equation.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.solve_burgers-Tuple{}","page":"Home","title":"DiscreteFiltering.solve_burgers","text":"solve_burgers()\n\nSolve Burgers equation.\n\n\n\n\n\n","category":"method"},{"location":"#DiscreteFiltering.solve_diffusion-Tuple{}","page":"Home","title":"DiscreteFiltering.solve_diffusion","text":"solve_diffusion()\n\nSolve diffusion equation.\n\n\n\n\n\n","category":"method"}]
}
