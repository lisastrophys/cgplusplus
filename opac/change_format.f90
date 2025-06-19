program test
implicit none
integer, PARAMETER :: DP = 16
real*8 :: abss(65), sca(65)
integer :: i, iloc(1)

open(100, file='dustopac_silicate.inp')
read(100, *)
do i=1,65
    read(100, *) abss(i)
    read(100, *)
enddo
read(100, *)
do i=1,65
    read(100, *) sca(i)
    read(100, *)
enddo
close(100)

open(101, file='dustopac_silicate_alt.inp')
do i=1,65
    write(101, '(E12.6E2)') abss(i)
enddo
write(101, *) ''

do i=1,65
    write(101, '(E12.6E2)') sca(i)
enddo
close(101)
end program
    
