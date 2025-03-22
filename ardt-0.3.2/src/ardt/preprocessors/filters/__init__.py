#  Copyright (c) 2024. Affects AI LLC
#
#  Licensed under the Creative Common CC BY-NC-SA 4.0 International License (the "License");
#  you may not use this file except in compliance with the License. The full text of the License is
#  provided in the included LICENSE file. If this file is not available, you may obtain a copy of the
#  License at
#
#       https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
#  Unless required by applicable law or agreed to in writing, software distributed under the License
#  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing permissions and limitations
#  under the License.

"""
This package contains preprocessors that perform signal filtering.

The AERTrial's load_signal_data method return an array of (N+1)xM where N is the number of signal channels, and M is
the length of the signal. The 0th row is timestep data, not signal data, and should be removed using a ChannelSelector
preprocessor prior to applying these filters.

Note - if you really don't care about the timestep data, leaving it inplace won't hurt anything, but there's no sense in
filtering it either.
"""
