use jxgl;
alter table sc drop foreign key sc_ibfk_3;
alter table sc drop foreign key sc_ibfk_4;
delimiter //
create trigger TR_SC_IN_SNO before insert on sc
for each row
begin
	if new.sno not in (select sno from student) then
		signal sqlstate 'HY000' set message_text = '该学号不存在！';
	end if;
end//
delimiter ;
insert into sno values('');
alter table sc add constraint sc_ibfc_1 foreign key (sno) references student(sno);
show triggers;