# Construct a list of all the ways in which the open indices of a tensor expression can be
# given a planar (cyclic) ordering. The list contains a representative for all inequivalent
# classes of planar orderings, where two orderings are equivalent if they are related by a
# cyclic permutation of the indices.
function get_possible_planar_indices(ex)
    if !TO.istensorexpr(ex)
        return [[]]
    elseif TO.isgeneraltensor(ex)
        _, leftind, rightind = TO.decomposegeneraltensor(ex)
        ind = planar_unique2(vcat(leftind, reverse(rightind)))
        return length(ind) == length(unique(ind)) ? Any[ind] : Any[]
    elseif isexpr(ex, :call) && (ex.args[1] == :+ || ex.args[1] == :-)
        inds = get_possible_planar_indices(ex.args[2])
        keep = fill(true, length(inds))
        for i in 3:length(ex.args)
            inds′ = get_possible_planar_indices(ex.args[i])
            keepᵢ = fill(false, length(inds))
            for (j, ind) in enumerate(inds), ind′ in inds′
                if iscyclicpermutation(ind′, ind)
                    keepᵢ[j] = true
                end
            end
            keep .&= keepᵢ
            any(keep) || break # give up early if keep is all false
        end
        return inds[keep]
    elseif isexpr(ex, :call) && ex.args[1] == :*
        length(ex.args) == 3 ||
            error("unexpected error occured: contraction should have been decomposed into tree by now")
        inds1 = get_possible_planar_indices(ex.args[2])
        inds2 = get_possible_planar_indices(ex.args[3])
        inds = Any[]
        for ind1 in inds1, ind2 in inds2
            for (oind1, oind2, cind1, cind2) in possible_planar_complements(ind1, ind2)
                push!(inds, vcat(oind1, oind2))
            end
        end
        return inds
    elseif isexpr(ex, :call) && ex.args[1] == :/
        return get_possible_planar_indices(ex.args[2])
    elseif isexpr(ex, :call) && ex.args[1] == :\
        return get_possible_planar_indices(ex.args[3])
    else
        return Any[]
    end
end

# Remove double indices (trace indices) from cyclic set, but only if the trace can be
# performed planarly. This requires that any two indices to be traced are only separated by
# other indices that are themselves also to be traced amongst each other. Hence, there is
# an innermost pair of indices to be traced, which can thus be removed first, and then the
# process can be repeated until no more indices can be traced.
function planar_unique2(allind)
    oind = collect(allind)
    removing = true
    while removing
        removing = false
        i = 1
        while i <= length(oind) && length(oind) > 1
            j = mod1(i + 1, length(oind))
            if oind[i] == oind[j]
                deleteat!(oind, i)
                deleteat!(oind, mod1(i, length(oind)))
                removing = true
            else
                i += 1
            end
        end
    end
    return oind
end

# Remove intersection (contraction indices) from two cyclic sets, again only if the
# contraction can be performed planarly. This requires that all contraction indices of both
# tensors are adjacent, and that they appear in opposite order in the two sets (which should
# be interpreted in the cyclic sense).
function possible_planar_complements(ind1, ind2)
    # quick return path
    (isempty(ind1) || isempty(ind2)) && return Any[(ind1, ind2, Any[], Any[])]
    # general case:
    j1 = findfirst(in(ind2), ind1)
    if j1 === nothing # disconnected diagrams, can be made planar in various ways
        return Any[
            (circshift(ind1, i - 1), circshift(ind2, j - 1), Any[], Any[])
                for i in eachindex(ind1), j in eachindex(ind2)
        ]
    else # genuine contraction
        N1, N2 = length(ind1), length(ind2)
        j2 = findfirst(==(ind1[j1]), ind2)
        jmax1 = j1
        jmin2 = j2
        while jmax1 < N1 && ind1[jmax1 + 1] == ind2[mod1(jmin2 - 1, N2)]
            jmax1 += 1
            jmin2 -= 1
        end
        jmin1 = j1
        jmax2 = j2
        if j1 == 1 && jmax1 < N1
            while ind1[mod1(jmin1 - 1, N1)] == ind2[mod1(jmax2 + 1, N2)]
                jmin1 -= 1
                jmax2 += 1
            end
        end
        if jmax2 > N2
            jmax2 -= N2
            jmin2 -= N2
        end
        indo1 = jmin1 < 1 ? ind1[(jmax1 + 1):mod1(jmin1 - 1, N1)] :
            vcat(ind1[(jmax1 + 1):N1], ind1[1:(jmin1 - 1)])
        cind1 = jmin1 < 1 ? vcat(ind1[mod1(jmin1, N1):N1], ind1[1:jmax1]) :
            ind1[jmin1:jmax1]
        indo2 = jmin2 < 1 ? ind2[(jmax2 + 1):mod1(jmin2 - 1, N2)] :
            vcat(ind2[(jmax2 + 1):N2], ind2[1:(jmin2 - 1)])
        cind2 = reverse(cind1)
        return isempty(intersect(indo1, indo2)) ? Any[(indo1, indo2, cind1, cind2)] : Any[]
    end
end
