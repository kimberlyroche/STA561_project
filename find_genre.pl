#!/usr/bin/perl
use strict;
use warnings;

# currently genre == action

my $desktop_path = "C:/Users/kim/Desktop";
my @files = glob("metadata/*.txt");

foreach my $file (@files) {
    if(-f $file) {
    	if($file =~ /^metadata\/(.*?)\.txt$/) {
			my $stub = $1;
			my $action = 0;
			open(DATA, $file);
			while(<DATA>) {
				if($_ =~ /^genre/) {
					if($_ =~ /^genre.*Action/) {
						print("ACTION: ".$stub."\n");
						system("cp posters/".$stub.".jpg ".$desktop_path."/action");
					} else {
						print("NO ACTION: ".$stub."\n");
						system("cp posters/".$stub.".jpg ".$desktop_path."/no_action");
					}
				}
			}
			close(DATA);
    	}
    }
}
