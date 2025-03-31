use temp;
set global log_bin_trust_function_creators = 1;
drop function if exists f;
drop procedure if exists p;
delimiter //
create function f(a int, b int) returns int
begin
	declare c int;
	set c = a;
    label: while 1 do
		set c = c * b;
        leave label;
    end while;
    return c;
end//

create procedure p(in a int, in b int, out c int)
	set c =  a * b;//
delimiter ;
set @a:=f(3, 2);
select @a;
set @a=0;
call p(1, 2, @a);