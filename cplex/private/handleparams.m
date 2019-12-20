function [newOptions,display]=handleparams(options)
%%
% Given an options set "options", this function creates a parameter structure
% newOptions to be passed to the toolbox optimization function invocation.
% It also sets up a display indicator if specified by the options.
%
%  See also cplexoptimset,
%

% ---------------------------------------------------------------------------
% File: handleparams.m
% ---------------------------------------------------------------------------
% Licensed Materials - Property of IBM
% 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
% Copyright IBM Corporation 2008, 2019. All Rights Reserved.
%
% US Government Users Restricted Rights - Use, duplication or
% disclosure restricted by GSA ADP Schedule Contract with
% IBM Corp.
% ---------------------------------------------------------------------------
