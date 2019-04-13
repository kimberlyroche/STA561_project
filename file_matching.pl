#!/usr/bin/perl
use strict;
use warnings;

my $desktop_path = "C:/Users/kim/Desktop";
my @files = glob("metadata/*.txt");

my $unmatched = 0;
foreach my $file (@files) {
    if(-f $file) {
    	if($file =~ /^metadata\/(.*?)\.txt$/) {
    		if(!(-e "posters/".$1.".jpg")) {
    			print($file." has no matching poster!\n");
    			$unmatched++;
    			system("mv ".$file." ".$desktop_path."/omitted_metadata");
    		}
    	}
    }
}
print("Total ".$unmatched." unmatched.\n");