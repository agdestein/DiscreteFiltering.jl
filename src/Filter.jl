abstract type Filter end


struct TopHatFilter <: Filter
    width::Function
end


